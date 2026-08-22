import { useState, useEffect } from 'react';
import { prismApi, type HealthResponse } from '../api/prismApi';

export function useHealth() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    prismApi.health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'error', ollama_reachable: false }))
      .finally(() => setLoading(false));

    const interval = setInterval(() => {
      prismApi.health()
        .then(setHealth)
        .catch(() => setHealth({ status: 'error', ollama_reachable: false }));
    }, 60_000);

    return () => clearInterval(interval);
  }, []);

  return { health, loading };
}
