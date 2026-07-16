import { HttpInterceptorFn } from '@angular/common/http';
import { inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

export const apiKeyInterceptor: HttpInterceptorFn = (req, next) => {
  if (!isPlatformBrowser(inject(PLATFORM_ID))) return next(req);
  const key = localStorage.getItem('fanta_api_key');
  if (!key) return next(req);
  return next(req.clone({ setHeaders: { 'X-API-Key': key } }));
};
