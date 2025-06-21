"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import "./App.css"
import ChatMessage from "./components/ChatMessage"
import ExampleQuestions from "./components/ExampleQuestions"
import type { Message } from "./types"

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm your immigration law assistant. Ask me any question about immigration law, and I'll try to answer based on the documents I've been trained on.",
      sources: [],
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Mock API response for frontend testing
  const mockApiResponse = (question: string) => {
    const responses = [
      {
        answer:
          "Based on the immigration law documents, work visas typically require: 1) A job offer from a qualified employer, 2) Labor certification in some cases, 3) Proof of qualifications and education, 4) Medical examination, and 5) Background checks. The specific requirements vary depending on the type of work visa (H-1B, L-1, O-1, etc.).",
        sources: ["Page 15", "Page 23", "Page 31"],
        processing_time: "1.2 seconds",
      },
      {
        answer:
          "Tourist visas generally allow stays of up to 90 days for most countries under the Visa Waiver Program. However, this can vary: B-2 tourist visas can be granted for up to 6 months initially, and extensions may be possible. The exact duration depends on your country of origin and the purpose of your visit.",
        sources: ["Page 8", "Page 12"],
        processing_time: "0.9 seconds",
      },
      {
        answer:
          "Family reunification processes vary depending on your relationship to the petitioner. Immediate relatives (spouses, unmarried children under 21, parents of US citizens) have no numerical limits. Other family members fall into preference categories with annual limits. The process typically involves filing Form I-130, waiting for approval, and then applying for an immigrant visa or adjustment of status.",
        sources: ["Page 45", "Page 52", "Page 67"],
        processing_time: "1.5 seconds",
      },
    ]

    // Return a random response or match based on keywords
    if (question.toLowerCase().includes("work") || question.toLowerCase().includes("visa")) {
      return responses[0]
    } else if (question.toLowerCase().includes("tourist") || question.toLowerCase().includes("stay")) {
      return responses[1]
    } else if (question.toLowerCase().includes("family") || question.toLowerCase().includes("reunification")) {
      return responses[2]
    } else {
      return {
        answer: `I understand you're asking about "${question}". Based on the legal documents I have access to, I can provide information about immigration law topics including work visas, tourist visas, family reunification, asylum processes, and permanent residency applications. Could you please rephrase your question to be more specific about which immigration topic you'd like to know about?`,
        sources: ["Page 1", "Page 5"],
        processing_time: "0.7 seconds",
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      sources: [],
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    const currentInput = input
    setInput("")
    setIsLoading(true)

    // Simulate API delay
    setTimeout(
      () => {
        const mockResponse = mockApiResponse(currentInput)

        // Add assistant message
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: mockResponse.answer,
          sources: mockResponse.sources,
          processingTime: mockResponse.processing_time,
          timestamp: new Date(),
        }

        setMessages((prev) => [...prev, assistantMessage])
        setIsLoading(false)
      },
      1000 + Math.random() * 2000,
    ) // Random delay between 1-3 seconds
  }

  const handleExampleClick = (question: string) => {
    setInput(question)
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Immigration Law Assistant</h1>
        <p>Ask questions about immigration law and get answers based on legal documents</p>
        <div className="demo-badge">
          <span>🧪 Frontend Demo Mode</span>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages-container">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {isLoading && (
            <ChatMessage
              message={{
                id: "loading",
                role: "assistant",
                content: "Thinking...",
                sources: [],
                timestamp: new Date(),
                isLoading: true,
              }}
            />
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about immigration law..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? "Thinking..." : "Ask"}
          </button>
        </form>

        <ExampleQuestions onExampleClick={handleExampleClick} />
      </main>

      <footer className="app-footer">
        <p>
          This assistant provides information based on specific legal documents. It is not a substitute for professional
          legal advice.
        </p>
      </footer>
    </div>
  )
}

export default App
