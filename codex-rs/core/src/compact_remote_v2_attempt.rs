//! Builds and sends foreground and speculative remote V2 compaction attempts.

use crate::context::UserGoalUpdate;
use codex_protocol::ResponseItemId;
use std::collections::HashSet;
use std::sync::Arc;

use super::CompactionRequestTransport;
use super::RemoteCompactionV2Output;
use super::run_remote_compaction_request_v2;
use crate::Prompt;
use crate::client::ModelClientSession;
use crate::compact::CompactionAnalyticsDetails;
use crate::compact_remote_history::trim_function_call_history_to_fit_context_window;
use crate::context::BaseInstructionsFragment;
use crate::context::world_state::WorldState;
use crate::context_manager::ContextManager;
use crate::context_manager::replace_unresolvable_image_files;
use crate::responses_metadata::CompactionTurnMetadata;
use crate::session::CompactionPrefix;
use crate::session::session::Session;
use crate::session::step_context::StepContext;
use codex_analytics::CompactionImplementation;
use codex_analytics::CompactionPhase;
use codex_analytics::CompactionReason;
use codex_analytics::CompactionTrigger;
use codex_history::CodexHarnessMetadata;
use codex_protocol::error::CodexErr;
use codex_protocol::error::CodexErrorDetails;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::TokenUsage;
use codex_rollout_trace::CompactionTraceContext;
use tokio_util::sync::CancellationToken;
use tracing::info;
use tracing::warn;

pub(super) struct RemoteCompactV2Attempt {
    pub(super) input_goal_ids: HashSet<ResponseItemId>,
    pub(super) trace_input_history: Option<Vec<ResponseItem>>,
    pub(super) prompt_input: Vec<ResponseItem>,
    pub(super) prompt_input_metadata: Vec<Option<CodexHarnessMetadata>>,
    pub(super) compaction_output: ResponseItem,
    pub(super) compaction_response_id: String,
    pub(super) token_usage: Option<TokenUsage>,
    /// Keeps a session created for standalone compaction alive through lifecycle completion.
    pub(super) owned_client_session: Option<ModelClientSession>,
}

pub(super) enum RemoteCompactV2AttemptInput<'a> {
    Foreground(Option<&'a mut ModelClientSession>),
    Background(BackgroundCompactionInput),
}

pub(crate) struct BackgroundCompactionCandidate {
    pub(crate) prefix: CompactionPrefix,
    pub(super) model_info: Arc<ModelInfo>,
    pub(super) cutoff_world_state: Arc<WorldState>,
    pub(super) attempt: RemoteCompactV2Attempt,
}

pub(crate) struct BackgroundCompactionInput {
    prefix: CompactionPrefix,
    turn_state: Option<String>,
    window: (String, u64, uuid::Uuid),
    base_instructions: BaseInstructions,
    input: Vec<ResponseItem>,
    metadata: Vec<Option<CodexHarnessMetadata>>,
}

pub(super) async fn run_remote_compact_v2_attempt(
    sess: &Arc<Session>,
    step_context: &Arc<StepContext>,
    compaction_trace: &CompactionTraceContext,
    compaction_metadata: CompactionTurnMetadata,
    analytics_details: &mut CompactionAnalyticsDetails,
    input: RemoteCompactV2AttemptInput<'_>,
) -> CodexResult<RemoteCompactV2Attempt> {
    let turn_context = &step_context.turn;
    let mut owned_client_session = None;
    let (
        model_info,
        input_goal_ids,
        mut trace_input_history,
        fallback_instructions,
        mut background_input,
        mut background_input_metadata,
        mut transport,
        captured_window,
        mut foreground_history,
    ) = match input {
        RemoteCompactV2AttemptInput::Foreground(client_session) => {
            let model_info = turn_context.model_info();
            let mut history = sess.clone_history().await;
            let input_goal_ids = UserGoalUpdate::message_ids(history.raw_items());
            let mut fallback_instructions = sess.get_prompt_base_instructions().await;
            if let Some(text) = history
                .raw_items()
                .find_map(BaseInstructionsFragment::text_from_item)
            {
                fallback_instructions.text = text.to_owned();
            } else if history.has_tool_declarations() {
                // An incremental window has no recorded base item when its instructions are empty.
                fallback_instructions.text.clear();
            }
            let tools_in_history = step_context.settings.model_info.use_responses_lite
                && history
                    .raw_items()
                    .any(|item| matches!(item, ResponseItem::AdditionalTools { .. }));
            let base_instructions = if tools_in_history {
                BaseInstructions {
                    text: String::new(),
                    provenance: None,
                }
            } else {
                fallback_instructions.clone()
            };
            let (rewritten_outputs, estimated_deleted_tokens) =
                trim_function_call_history_to_fit_context_window(
                    &mut history,
                    turn_context,
                    turn_context.model_context_window(),
                    &base_instructions,
                );
            if rewritten_outputs > 0 {
                info!(
                    turn_id = %turn_context.sub_id,
                    rewritten_outputs,
                    "rewrote history outputs before remote compaction v2"
                );
            }
            if estimated_deleted_tokens > 0 {
                let max_local_deleted_tokens = sess
                    .estimated_tokens_after_last_model_generated_item()
                    .await;
                analytics_details.active_context_tokens_before = analytics_details
                    .active_context_tokens_before
                    .map(|active_context_tokens_before| {
                        active_context_tokens_before
                            .saturating_sub(estimated_deleted_tokens.min(max_local_deleted_tokens))
                    });
            }
            let trace_input_history = compaction_trace
                .is_enabled()
                .then(|| history.raw_items().cloned().collect());
            let client_session = client_session.unwrap_or_else(|| {
                owned_client_session.insert(sess.services.model_client.new_session())
            });
            (
                model_info,
                input_goal_ids,
                trace_input_history,
                fallback_instructions,
                Vec::new(),
                Vec::new(),
                CompactionRequestTransport::Foreground(client_session),
                None,
                Some(history),
            )
        }
        RemoteCompactV2AttemptInput::Background(input) => (
            &step_context.settings.model_info,
            UserGoalUpdate::message_ids(input.input.iter()),
            None,
            input.base_instructions,
            input.input,
            input.metadata,
            CompactionRequestTransport::Background(
                input.turn_state,
                input.prefix.auth_owner_generation,
            ),
            Some(input.window),
            None,
        ),
    };
    let tool_router = &step_context.tool_router;
    let mut responses_metadata = match captured_window {
        Some(window) => {
            sess.compaction_responses_metadata_for_window(turn_context, compaction_metadata, window)
        }
        None => {
            sess.compaction_responses_metadata(turn_context, compaction_metadata)
                .await
        }
    };
    responses_metadata.tool_namespaces_info = tool_router.tool_namespaces_info().cloned();
    loop {
        let (prompt_input, prompt_input_metadata): (Vec<_>, Vec<_>) =
            if let Some(history) = foreground_history.as_ref() {
                let (mut prompt_input, prompt_input_metadata): (Vec<_>, Vec<_>) = history
                    .clone()
                    .for_prompt_annotated(
                        &model_info.input_modalities,
                        sess.services.image_store.as_ref(),
                    )
                    .await
                    .into_iter()
                    .map(|envelope| (envelope.item, envelope.metadata))
                    .unzip();
                sess.services
                    .executed_tool_calls
                    .attach_to_compaction_prompt(&mut prompt_input);
                (prompt_input, prompt_input_metadata)
            } else {
                (
                    std::mem::take(&mut background_input),
                    std::mem::take(&mut background_input_metadata),
                )
            };
        let tools_in_history = model_info.use_responses_lite
            && prompt_input
                .iter()
                .any(|item| matches!(item, ResponseItem::AdditionalTools { .. }));
        let instructions_in_history = model_info.use_responses_lite
            && prompt_input
                .iter()
                .any(|item| BaseInstructionsFragment::text_from_item(item).is_some());
        let prompt_base_instructions = if instructions_in_history {
            BaseInstructions {
                text: String::new(),
                provenance: None,
            }
        } else {
            fallback_instructions.clone()
        };
        let mut request_input = prompt_input;
        request_input.push(ResponseItem::CompactionTrigger {});
        let prompt = Prompt {
            input: request_input,
            tools: if tools_in_history {
                Arc::default()
            } else {
                tool_router.model_visible_specs()
            },
            parallel_tool_calls: true,
            base_instructions: prompt_base_instructions,
            output_schema: None,
            output_schema_strict: true,
            cyber_access_program: turn_context.cyber_access_program,
        };
        let trace_attempt = compaction_trace.start_attempt(&serde_json::json!({
            "model": model_info.slug.as_str(),
            "instructions": prompt.base_instructions.text.as_str(),
            "input": &prompt.input,
            "parallel_tool_calls": prompt.parallel_tool_calls,
        }));
        let compaction_output_result = run_remote_compaction_request_v2(
            sess,
            step_context,
            &prompt,
            &responses_metadata,
            &mut transport,
        )
        .await;
        trace_attempt.record_result(
            compaction_output_result
                .as_ref()
                .map(|output| std::slice::from_ref(&output.compaction_output)),
        );
        match compaction_output_result {
            Ok(RemoteCompactionV2Output {
                compaction_output,
                response_id,
                token_usage,
            }) => {
                if compaction_trace.is_enabled() {
                    trace_input_history = foreground_history
                        .as_ref()
                        .map(|history| history.raw_items().cloned().collect());
                }
                let mut prompt_input = prompt.input;
                prompt_input.pop();
                return Ok(RemoteCompactV2Attempt {
                    input_goal_ids,
                    trace_input_history,
                    prompt_input,
                    prompt_input_metadata,
                    compaction_output,
                    compaction_response_id: response_id,
                    token_usage,
                    owned_client_session,
                });
            }
            Err(err)
                if matches!(err.details(), CodexErrorDetails::ContextWindowExceeded)
                    && foreground_history
                        .as_ref()
                        .is_some_and(|history| history.raw_items().len() > 1) =>
            {
                warn!(
                    turn_id = %turn_context.sub_id,
                    error = %err,
                    "remote compaction exceeded the context window; removing oldest history item"
                );
                if let Some(history) = foreground_history.as_mut() {
                    history.remove_first_item();
                }
            }
            Err(err) => return Err(err),
        }
    }
}

// Defer normalization and attribution until the caller releases the session lock.
pub(crate) async fn capture_background_input(
    sess: &Session,
    step_context: &StepContext,
    mut snapshot: ContextManager,
    base_instructions: BaseInstructions,
    turn_state: Option<String>,
    auth_owner_generation: Option<u64>,
    window: (String, u64, uuid::Uuid),
) -> BackgroundCompactionInput {
    let base_instructions = if step_context.settings.model_info.use_responses_lite
        && snapshot
            .raw_items()
            .any(|item| matches!(item, ResponseItem::AdditionalTools { .. }))
    {
        BaseInstructions {
            text: String::new(),
            provenance: None,
        }
    } else {
        base_instructions
    };
    let prefix = CompactionPrefix {
        generation: snapshot.history_version(),
        len: snapshot.annotated_items().len(),
        auth_owner_generation,
    };
    trim_function_call_history_to_fit_context_window(
        &mut snapshot,
        &step_context.turn,
        step_context.settings.model_info.usable_context_window(),
        &base_instructions,
    );
    snapshot.normalize_history(&step_context.settings.model_info.input_modalities);
    let (mut input, metadata): (Vec<_>, Vec<_>) = snapshot
        .into_annotated_items()
        .into_iter()
        .map(|envelope| (envelope.item, envelope.metadata))
        .unzip();
    sess.services
        .executed_tool_calls
        .attach_to_compaction_prompt(&mut input);
    BackgroundCompactionInput {
        prefix,
        turn_state,
        window,
        base_instructions,
        input,
        metadata,
    }
}

pub(crate) async fn run_background_compaction(
    sess: &Arc<Session>,
    step_context: Arc<StepContext>,
    cutoff_world_state: Arc<WorldState>,
    phase: CompactionPhase,
    mut prepared_input: BackgroundCompactionInput,
    shutdown_cancellation: &CancellationToken,
) -> CodexResult<BackgroundCompactionCandidate> {
    tokio::select! {
        _ = shutdown_cancellation.cancelled() => return Err(CodexErr::TurnAborted),
        () = replace_unresolvable_image_files(
            prepared_input.input.iter_mut(),
            sess.services.image_store.as_ref(),
        ) => {},
    }
    let prefix = prepared_input.prefix;
    let metadata = CompactionTurnMetadata::new(
        CompactionTrigger::Auto,
        CompactionReason::ContextLimit,
        CompactionImplementation::ResponsesCompactionV2,
        phase,
    );
    let attempt = run_remote_compact_v2_attempt(
        sess,
        &step_context,
        &CompactionTraceContext::disabled(),
        metadata,
        &mut CompactionAnalyticsDetails::default(),
        RemoteCompactV2AttemptInput::Background(prepared_input),
    )
    .await?;
    Ok(BackgroundCompactionCandidate {
        prefix,
        model_info: Arc::clone(&step_context.settings.model_info),
        cutoff_world_state,
        attempt,
    })
}
