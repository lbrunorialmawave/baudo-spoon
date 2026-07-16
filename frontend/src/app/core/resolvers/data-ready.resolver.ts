import { inject } from '@angular/core';
import { ResolveFn } from '@angular/router';
import { catchError, of } from 'rxjs';
import { IntelligenceService } from '../services/intelligence.service';
import { ClusteringResponse } from '../models/api.models';

export const dataReadyResolver: ResolveFn<ClusteringResponse | null> = () => {
  const intelligenceService = inject(IntelligenceService);

  return intelligenceService.getClusters().pipe(
    catchError(() => of(null))
  );
};
