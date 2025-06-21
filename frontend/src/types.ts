export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  sources: string[]
  timestamp: Date
  processingTime?: string
  isLoading?: boolean
  isError?: boolean
}
