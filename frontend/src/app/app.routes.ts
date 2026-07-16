import { Routes } from '@angular/router';
import { apiKeyGuard } from './core/guards/api-key.guard';
import { dataReadyResolver } from './core/resolvers/data-ready.resolver';
import { ShellComponent } from './shared/components/shell/shell.component';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'dashboard',
    pathMatch: 'full',
  },
  {
    path: '',
    component: ShellComponent,
    canActivate: [apiKeyGuard],
    children: [
      {
        path: 'dashboard',
        resolve: { clusterData: dataReadyResolver },
        loadComponent: () =>
          import('./features/dashboard/dashboard.component').then(m => m.DashboardComponent),
      },
      {
        path: 'players',
        loadComponent: () =>
          import('./features/players/players.component').then(m => m.PlayersComponent),
      },
      {
        path: 'quotations',
        loadComponent: () =>
          import('./features/quotations/quotations.component').then(m => m.QuotationsComponent),
      },
      {
        path: 'matches',
        loadComponent: () =>
          import('./features/matches/matches.component').then(m => m.MatchesComponent),
      },
      {
        path: 'predictions',
        loadComponent: () =>
          import('./features/predictions/predictions.component').then(m => m.PredictionsComponent),
      },
      {
        path: 'optimizer',
        loadComponent: () =>
          import('./features/optimizer/optimizer.component').then(m => m.OptimizerComponent),
      },
      {
        path: 'auction',
        loadComponent: () =>
          import('./features/auction/auction.component').then(m => m.AuctionComponent),
      },
    ],
  },
  {
    path: 'setup',
    loadComponent: () =>
      import('./features/setup/setup.component').then(m => m.SetupComponent),
  },
  {
    path: '**',
    loadComponent: () =>
      import('./features/not-found/not-found.component').then(m => m.NotFoundComponent),
  },
];
