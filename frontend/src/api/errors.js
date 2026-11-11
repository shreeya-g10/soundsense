// Thrown by every API function (real and mock) on a non-2xx response.
// `detail` is the backend's {"detail": "..."} message; `status` is the HTTP status.
export class ApiError extends Error {
  constructor(status, detail) {
    super(`${status}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}
