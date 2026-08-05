import { HttpErrorResponse, HttpHandlerFn, HttpInterceptorFn, HttpRequest } from '@angular/common/http';
import { inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { catchError, throwError } from 'rxjs';

import { HealthCheckService } from '../services/health-check.service';

const UNREACHABLE_STATUSES = new Set([0, 502, 503, 504]);

/** On a backend-unreachable error from any call, recheck /health and surface the full-screen gate if it's still down. */
export const healthCheckInterceptor: HttpInterceptorFn = (req: HttpRequest<unknown>, next: HttpHandlerFn) => {
  if (!isPlatformBrowser(inject(PLATFORM_ID))) return next(req);
  if (req.url.includes('/health')) return next(req);

  const health = inject(HealthCheckService);

  return next(req).pipe(
    catchError((err: unknown) => {
      if (err instanceof HttpErrorResponse && UNREACHABLE_STATUSES.has(err.status)) {
        health.recheck();
      }
      return throwError(() => err);
    }),
  );
};
