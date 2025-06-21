'use client'
import './ChatOptions.css'
import { useState } from 'react'

interface ChatOptionsProps {
  onOptionSelect: (value: string) => void;
}

const ChatOptions: React.FC<ChatOptionsProps> = ({ onOptionSelect }) => {
  const [dropdownValue, setDropdownValue] = useState('');

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    console.log(value);
    setDropdownValue(value);
    onOptionSelect(value);  // <-- Lift state up
  };

  return (
    <div className="chat-options-container">
      <label htmlFor="chat-select">Select an option</label>
      <select
        id="chat-select"
        className="chat-options-select"
        value={dropdownValue}
        onChange={handleSelectChange}
      >
        <option value="immigration law">General questions</option>
        <option value="Asylum (I-589)">Asylum (I-589)</option>
        <option value="NIW">I-140 (NIW)</option>
      </select>
    </div>
  );
};

export default ChatOptions;
