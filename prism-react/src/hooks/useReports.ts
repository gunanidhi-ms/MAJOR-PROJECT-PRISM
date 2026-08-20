import { useState, useEffect, useCallback } from 'react';
import { prismApi, type Report, type ReportsListResponse } from '../api/prismApi';

export function useReports() {
  const [data, setData] = useState<ReportsListResponse>({ total: 0, reports: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReports = useCallback(async () => {
    setLoading(true);
    try {
      const result = await prismApi.listReports();
      setData(result);
      setError(result.error || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reports');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchReports(); }, [fetchReports]);

  return { data, loading, error, refetch: fetchReports };
}

export function useReport(studyId: string) {
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!studyId) return;
    setLoading(true);
    prismApi.getReport(studyId)
      .then(setReport)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load report'))
      .finally(() => setLoading(false));
  }, [studyId]);

  return { report, loading, error, setReport };
}
