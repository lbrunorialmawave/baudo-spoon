export interface MetricPoint {
  run_id: string;
  trained_at: string;
  model_name: string;
  status: 'ok' | 'degraded' | 'error';
  metric_value: number;
}

export interface MetricEntry {
  metric: string;
  value: number;
  split: string;
}

export interface ModelRun {
  run_id: string;
  model_name: string;
  trained_at: string;
  season_start: number | null;
  git_commit: string | null;
  status: 'ok' | 'degraded' | 'error';
  metrics: MetricEntry[];
}

export interface ModelRunsResponse {
  items: ModelRun[];
  offset: number;
  limit: number;
}

export interface CompareResponse {
  run_a: ModelRun | null;
  run_b: ModelRun | null;
}
