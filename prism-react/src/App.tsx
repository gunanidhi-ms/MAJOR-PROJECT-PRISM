import { lazy, Suspense, useEffect } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { PatientSessionProvider } from './context/PatientSessionContext';

// Code-split every page — they load only when first visited
const Dashboard    = lazy(() => import('./pages/Dashboard'));
const Generate     = lazy(() => import('./pages/Generate'));
const ReportsList  = lazy(() => import('./pages/ReportsList'));
const ReportDetail = lazy(() => import('./pages/ReportDetail'));
// @ts-ignore
const LiveViewer   = lazy(() => import('./pages/LiveViewer'));


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
  useEffect(() => {
    // 1. Cancel any pending shutdown on page load (e.g. after refresh)
    fetch('http://localhost:8001/api/system/cancel-shutdown', { method: 'POST' }).catch(() => {});

    // 2. Schedule shutdown when the tab/browser is closed (hit 'X')
    const handleUnload = () => {
      navigator.sendBeacon('http://localhost:8001/api/system/shutdown?grace=2.5');
    };

    window.addEventListener('pagehide', handleUnload);
    window.addEventListener('beforeunload', handleUnload);

    return () => {
      window.removeEventListener('pagehide', handleUnload);
      window.removeEventListener('beforeunload', handleUnload);
    };
  }, []);

  return (
    <PatientSessionProvider>
      <BrowserRouter>
        <Layout>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/"                 element={<Dashboard />} />
              <Route path="/generate"         element={<Generate />} />
              <Route path="/reports"          element={<ReportsList />} />
              <Route path="/report/:studyId"  element={<ReportDetail />} />
              <Route path="/live"             element={<LiveViewer />} />
              <Route path="*"                 element={<Dashboard />} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </PatientSessionProvider>
  );
}
