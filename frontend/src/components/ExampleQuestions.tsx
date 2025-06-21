"use client"

import type React from "react"
import "./ExampleQuestions.css"

interface ExampleQuestionsProps {
  onExampleClick: (question: string) => void
}

const ExampleQuestions: React.FC<ExampleQuestionsProps> = ({ onExampleClick }) => {
  const examples = [
    "What are the requirements for a work visa?",
    "How long can I stay in the country with a tourist visa?",
    "What is the process for family reunification?",
    "What are the grounds for asylum?",
    "How can I apply for permanent residency?",
  ]

  return (
    <div className="example-questions">
      <p className="examples-title">Try asking:</p>
      <div className="examples-list">
        {examples.map((question, index) => (
          <button key={index} className="example-button" onClick={() => onExampleClick(question)}>
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}

export default ExampleQuestions
