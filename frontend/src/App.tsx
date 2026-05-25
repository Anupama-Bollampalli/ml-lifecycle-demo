import { useState } from 'react'
import ModelComparison from './components/ModelComparison'
import PredictionPanel from './components/PredictionPanel'
import ConfusionMatrix from './components/ConfusionMatrix'
import FeatureImportance from './components/FeatureImportance'
import LearningCurve from './components/LearningCurve'

type Tab = 'compare' | 'predict' | 'confusion' | 'features' | 'learning'

const TABS: { id: Tab; label: string }[] = [
  { id: 'compare', label: 'Compare Models' },
  { id: 'predict', label: 'Predict' },
  { id: 'confusion', label: 'Confusion Matrix' },
  { id: 'features', label: 'Feature Importance' },
  { id: 'learning', label: 'Learning Curves' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('compare')

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-4">
          <h1 className="text-2xl font-bold text-slate-800">ML Lifecycle Demo</h1>
          <span className="bg-indigo-100 text-indigo-700 text-xs font-semibold px-3 py-1 rounded-full">
            Breast Cancer Dataset
          </span>
        </div>
        {/* Tabs */}
        <div className="max-w-7xl mx-auto px-4">
          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'compare' && <ModelComparison />}
        {activeTab === 'predict' && <PredictionPanel />}
        {activeTab === 'confusion' && <ConfusionMatrix />}
        {activeTab === 'features' && <FeatureImportance />}
        {activeTab === 'learning' && <LearningCurve />}
      </main>
    </div>
  )
}
