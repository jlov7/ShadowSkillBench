import workbenchFixture from "@/data/workbench.json";
import practiceFixture from "@/data/practice.fixture.json";

export type WorkbenchFixture = {
  schema: string;
  schema_version: string;
  read_only: boolean;
  claim_ceiling: {
    status: "hold" | "confirmatory";
    descriptive_only: boolean;
    message: string;
    technical_exclusions?: {
      count: number;
      approved: boolean;
      max_total: number | null;
      max_per_primary_cell: number | null;
    };
  };
  protocol: {
    protocol_tag: string;
    code_commit: string;
    freeze_manifest_sha256: string | null;
    analysis_dataset_sha256: string | null;
    external_anchor_locator: string | null;
    anchor_verified: boolean;
    experiment_audit_status: string;
  };
  figures: Array<{
    id: string;
    title: string;
    summary: string;
    columns: string[];
    rows: Array<Record<string, unknown>>;
  }>;
  summaries: {
    cost: { included_episodes: number };
    metrics: {
      ordinary_task_completion: WorkbenchMetric;
      completion_under_policy: WorkbenchMetric;
      unsafe_imitation: WorkbenchMetric;
      false_enforcement: WorkbenchMetric;
      instruction_hierarchy_delta: WorkbenchMetric;
      latency: WorkbenchMetric;
      human_review_burden: WorkbenchMetric;
    };
    domains: Array<{
      domain: string;
      included_episodes: number;
      completion_under_policy_rate: number | null;
      unsafe_imitation_rate: number | null;
    }>;
    representative_traces: unknown[];
  };
  episode_index: Array<Record<string, unknown>>;
  episode_traces: PublicEpisodeTrace[];
  analysis: {
    raw_counts: Array<Record<string, unknown>>;
    estimands: Array<Record<string, unknown>>;
    average_marginal_effects: Array<Record<string, unknown>>;
    stage_b: {
      e6_unsafe_imitation: Array<Record<string, unknown>>;
      e7_false_enforcement: Array<Record<string, unknown>>;
      e8_authority_aware_gain: Array<Record<string, unknown>>;
      e9_completion_under_policy_delta: Array<Record<string, unknown>>;
      e9_unsafe_imitation_delta: Array<Record<string, unknown>>;
    };
  };
  claims: { ledger_valid: boolean; renderable_finding_ids: string[] };
  limitations: string[];
  content_hash: string;
};

export type WorkbenchMetric = {
  status: "available" | "unavailable";
  unit: string;
  value: number | null;
  numerator?: number;
  denominator?: number;
  reason?: string;
};

export type PublicEpisodeTrace = {
  episode_id: string;
  trace_artifact_sha256: string;
  events: Array<{
    sequence: number;
    kind: string;
    summary: string;
    role?: string;
    tool_name?: string;
    local_status?: string;
    reason_code?: string;
  }>;
};

export type PracticeFixture = {
  label: string;
  notice: string;
  policy_hash: string;
  demonstrations: Array<Record<string, string | number | null>>;
  skills: {
    skill_ir: string;
    rendered_skill: string;
    provenance: Array<{ label: string; value: string }>;
    risk_label: string;
    risk_summary: string;
  };
  policy: {
    case_id: string;
    columns: Array<{
      name: string;
      visible_context: string;
      tool_sequence: string;
      completion_claim: string;
      cup: string;
      token_cost: string;
    }>;
    identity_note: string;
  };
  authority: {
    nodes: Array<{ class_name?: string; label: string; note?: string }>;
    edges: Array<{ label: string; meaning: string }>;
    paths: string[];
  };
  episodes: {
    left: PracticeEpisode;
    right: PracticeEpisode;
  };
};

export type PracticeEpisode = {
  id: string;
  title: string;
  events: Array<{ kind: string; summary: string }>;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseWorkbenchDataset(value: unknown): WorkbenchFixture {
  if (!isRecord(value) || value.schema !== "SSB-WORKBENCH-1" || value.schema_version !== "1.0") {
    throw new Error("Invalid ShadowSkillBench workbench dataset schema");
  }
  if (value.read_only !== true || typeof value.content_hash !== "string") {
    throw new Error("Workbench dataset must be a read-only content-hashed export");
  }
  for (const field of ["claim_ceiling", "protocol", "preregistration", "analysis", "figures", "summaries", "episode_index", "episode_traces", "claims", "limitations"]) {
    if (!(field in value)) throw new Error(`Workbench dataset is missing ${field}`);
  }
  return value as unknown as WorkbenchFixture;
}

export const workbenchData = parseWorkbenchDataset(workbenchFixture);
export const practiceData = practiceFixture as PracticeFixture;
export const isHoldFixture = workbenchData.claim_ceiling.status === "hold";

export function unavailable(value: string | null | undefined): string {
  return value && value !== "unavailable" ? value : "Unavailable";
}
