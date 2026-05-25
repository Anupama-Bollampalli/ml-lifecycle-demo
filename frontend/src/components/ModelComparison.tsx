import { useEffect, useState } from 'react'
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

type ModelMetrics = {
  accuracy: number
  precision: number
  recall: number
  f1: number
  auc_roc: number
  training_time: number
}

type ComparisonData = Record<string, ModelMetrics>

const METRIC_KEYS: (keyof ModelMetrics)[] = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
const METRIC_LABELS: Record<string, string> = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1: 'F1 Score',
  auc_roc: 'AUC-ROC',
}

const MODEL_COLORS: Record<string, string> = {
  logistic: '#6366f1',
  forest: '#10b981',
  neural: '#f59e0b',
}

const MODEL_DISPLAY: Record<string, string> = {
  logistic: 'Logistic Regression',
  forest: 'Random Forest',
  neural: 'Neural Network',
}

export default function ModelComparison() {
  const [data, setData] = useState<ComparisonData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/models/comparison')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-slate-500 py-12 text-center">Loading metrics...</div>
  if (error) return <div className="text-red-500 py-12 text-center">Error: {error}</div>
  if (!data) return null

  const modelNames = Object.keys(data)

  // Find best value per metric
  const bestPerMetric: Record<string, number> = {}
  for (const metric of METRIC_KEYS) {
    bestPerMetric[metric] = Math.max(...modelNames.map((m) => data[m][metric] ?? 0))
  }

  // Build radar chart data
  const radarData = METRIC_KEYS.map((key) => {
    const entry: Record<string, string | number> = { metric: METRIC_LABELS[key] }
    for (const m of modelNames) {
      entry[m] = parseFloat(((data[m][key] ?? 0) * 100).toFixed(2))
    }
    return entry
  })

  return (
    <div className="space-y-8">
      <h2 className="text-xl font-semibold text-slate-700">Model Comparison</h2>

      {/* Metrics Table */}
      <div className="bg-white rounded-xl shadow overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-100 text-slate-600">
              <th className="text-left px-6 py-3 font-semibold">Metric</th>
              {modelNames.map((m) => (
                <th key={m} className="text-center px-6 py-3 font-semibold">
                  {MODEL_DISPLAY[m] ?? m}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {METRIC_KEYS.map((metric, idx) => (
              <tr key={metric} className={idx % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                <td className="px-6 py-3 font-medium text-slate-700">{METRIC_LABELS[metric]}</td>
                {modelNames.map((m) => {
                  const val = data[m][metric] ?? 0
                  const isBest = Math.abs(val - bestPerMetric[metric]) < 1e-9
                  return (
                    <td
                      key={m}
                      className={`text-center px-6 py-3 font-mono ${
                        isBest
                          ? 'bg-green-100 text-green-800 font-bold rounded'
                          : 'text-slate-600'
                      }`}
                    >
                      {(val * 100).toFixed(2)}%
                    </td>
                  )
                })}
              </tr>
            ))}
            <tr className="bg-slate-50 border-t border-slate-200">
              <td className="px-6 py-3 font-medium text-slate-700">Training Time</td>
              {modelNames.map((m) => (
                <td key={m} className="text-center px-6 py-3 font-mono text-slate-600">
                  {data[m].training_time?.toFixed(3) ?? '—'}s
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {/* Radar Chart */}
      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-lg font-semibold text-slate-700 mb-4">Performance Radar</h3>
        <ResponsiveContainer width="100%" height={350}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 13 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 11 }} />
            {modelNames.map((m) => (
              <Radar
                key={m}
                name={MODEL_DISPLAY[m] ?? m}
                dataKey={m}
                stroke={MODEL_COLORS[m] ?? '#888'}
                fill={MODEL_COLORS[m] ?? '#888'}
                fillOpacity={0.15}
                strokeWidth={2}
              />
            ))}
            <Legend />
            <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
