export type StepType =
  | 'start'
  | 'retrieve'
  | 'grade'
  | 'rewrite'
  | 'generate'
  | 'hallucination_check'
  | 'final'
  | 'error'

export interface SSEEvent {
  type: StepType
  message: string
  data?: Record<string, unknown>
  timestamp: string
  session_id?: string
  query_id?: string
}

export interface GradedDocument {
  id: string
  content: string
  source: string
  relevance_score: number
  grade: 'relevant' | 'irrelevant' | 'ungraded'
  grade_reason: string
}

export interface FinalAnswer {
  answer: string
  confidence: number
  retry_count: number
  hallucination_detected: boolean
  sources: Array<{ source: string; grade: string }>
  query_id: string
}
