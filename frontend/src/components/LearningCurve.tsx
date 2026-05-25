import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

type ModelKey = 'logistic' | 'forest' | 'neural'

const MODEL_OPTIONS: { key: ModelKey; label: string }[] = [
  { key: 'logistic', label: 'Logistic Regression' },
  { key: 'forest', label: 'Random Forest' },
  { key: 'neural', label: 'Neural Network' },
]

type CurvePoint = { epoch: number; train_loss: number; val_loss: number }

export default function LearningCurve() {
  const [selectedModel, setSelectedModel] = useState<ModelKey>('neural')
  const [curve, setCurve] = useState<CurvePoint[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch(`/api/models/${selectedModel}/learning-curve`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d) => setCurve(d.learning_curve ?? []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedModel])

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-700">Learning Curves</h2>

      <div className="bg-white rounded-xl shadow p-6 space-y-5">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">Select Model</label>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value as ModelKey)}
            className="border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            {MODEL_OPTIONS.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
          </select>
        </div>

        {loading && <div className="text-slate-400 text-sm py-8 text-center">Loading...</div>}
        {error && <div className="text-red-500 text-sm">Error: {error}</div>}

        {!loading && curve.length > 0 && (
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={curve} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="epoch"
                label={{ value: 'Epoch / Step', position: 'insideBottom', offset: -4, fontSize: 12 }}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 12 }}
                tick={{ fontSize: 11 }}
                tickFormatter={(v: number) => v.toFixed(3)}
              />
              <Tooltip formatter={((v: number) => v.toFixed(4)) as any} />
              <Legend wrapperStyle={{ paddingTop: '12px' }} />
              <Line
                type="monotone"
                dataKey="train_loss"
                name="Train Loss"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                name="Validation Loss"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                strokeDasharray="5 5"
                activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}

        {!loading && curve.length === 0 && !error && (
          <div className="text-slate-400 text-sm text-center py-8">No data available.</div>
        )}
      </div>

      <p className="text-xs text-slate-400">
        Note: For Logistic Regression and Random Forest, learning curves are synthetic approximations showing convergence behavior.
        For Neural Network, curves reflect actual epoch-by-epoch training loss.
      </p>
    </div>
  )
}
