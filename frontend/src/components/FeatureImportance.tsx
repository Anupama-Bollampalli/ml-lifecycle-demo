import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'

type FeatureEntry = { name: string; importance: number }

function getBarColor(rank: number, total: number): string {
  const pct = rank / total
  if (pct < 0.25) return '#6366f1' // indigo - top 25%
  if (pct < 0.5) return '#06b6d4'  // cyan
  if (pct < 0.75) return '#64748b' // slate
  return '#9ca3af'                  // gray
}

const API = import.meta.env.VITE_API_URL ?? ''

export default function FeatureImportance() {
  const [features, setFeatures] = useState<FeatureEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API}/features/importance`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d) => setFeatures(d.features ?? []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-slate-400 py-12 text-center">Loading features...</div>
  if (error) return <div className="text-red-500 py-12 text-center">Error: {error}</div>

  const chartData = features.map((f) => ({
    name: f.name,
    importance: parseFloat((f.importance * 100).toFixed(3)),
  }))

  const total = chartData.length

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-700">Feature Importance (Random Forest)</h2>

      <div className="bg-white rounded-xl shadow p-6">
        <div className="flex gap-4 mb-4 text-xs">
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-indigo-500 inline-block" /> Top 25%</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-cyan-500 inline-block" /> 25–50%</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-slate-500 inline-block" /> 50–75%</span>
          <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm bg-gray-400 inline-block" /> Bottom 25%</span>
        </div>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ left: 140, right: 20, top: 5, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              tick={{ fontSize: 11 }}
              width={135}
            />
            <Tooltip
              formatter={((v: number) => [`${v.toFixed(3)}%`, 'Importance']) as any}
            />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {chartData.map((_, idx) => (
                <Cell key={idx} fill={getBarColor(idx, total)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
