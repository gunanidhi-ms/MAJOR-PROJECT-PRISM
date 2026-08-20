import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';

// Code-split every page — they load only when first visited
const Dashboard    = lazy(() => import('./pages/Dashboard'));
const Generate     = lazy(() => import('./pages/Generate'));
const ReportsList  = lazy(() => import('./pages/ReportsList'));
const ReportDetail = lazy(() => import('./pages/ReportDetail'));


function PageLoader() {
  return (
    <div style={{ padding: '48px 0', display: 'flex', flexDirection: 'column', gap: 16 }}>
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="skeleton"
          style={{ height: i === 1 ? 200 : 80, borderRadius: 14, animationDelay: `${i * 0.1}s` }}
        />
      ))}
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/"                 element={<Dashboard />} />
            <Route path="/generate"         element={<Generate />} />
            <Route path="/reports"          element={<ReportsList />} />
            <Route path="/report/:studyId"  element={<ReportDetail />} />
            <Route path="*"                 element={<Dashboard />} />
          </Routes>
        </Suspense>
      </Layout>
    </BrowserRouter>
  );
}
