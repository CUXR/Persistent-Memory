export interface AppError {
  message: string;
  retryable: boolean;
  cause?: unknown;
}
