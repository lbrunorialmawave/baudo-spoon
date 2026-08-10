import { HttpInterceptorFn, HttpRequest, HttpHandlerFn, HttpErrorResponse } from '@angular/common/http';
import { inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { catchError, switchMap, throwError } from 'rxjs';
import { AuthService } from '../services/auth.service';
import { Router } from '@angular/router';

export const apiKeyInterceptor: HttpInterceptorFn = (req: HttpRequest<unknown>, next: HttpHandlerFn) => {
  if (!isPlatformBrowser(inject(PLATFORM_ID))) return next(req);

  // Skip auth endpoints to avoid infinite refresh loops
  if (req.url.includes('/auth/')) return next(req);

  const auth = inject(AuthService);
  const router = inject(Router);
  const token = auth.getAccessToken();

  const authed = token
    ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
    : req;

  return next(authed).pipe(
    catchError((err: unknown) => {
      if (err instanceof HttpErrorResponse && err.status === 401) {
        return auth.refresh().pipe(
          switchMap(() => {
            const newToken = auth.getAccessToken();
            const retried = newToken
              ? req.clone({ setHeaders: { Authorization: `Bearer ${newToken}` } })
              : req;
            return next(retried);
          }),
          catchError(() => {
            // Il refresh token non è più valido (scaduto o reuse detection
            // lato server): non ha più senso tenerlo in localStorage.
            auth.clearSession();
            router.navigate(['/setup']);
            return throwError(() => err);
          })
        );
      }
      return throwError(() => err);
    })
  );
};
