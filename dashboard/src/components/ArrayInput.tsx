import React, { useState, useRef, KeyboardEvent } from "react";
import { X, Plus } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface ArrayInputProps {
  value: string[] | number[];
  onChange: (value: string[] | number[]) => void;
  isNumeric?: boolean;
  placeholder?: string;
}

export function ArrayInput({ 
  value = [], 
  onChange, 
  isNumeric = false, 
  placeholder = "Add item..." 
}: ArrayInputProps) {
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const handleAddItem = () => {
    if (!inputValue.trim()) return;
    
    const newValue = isNumeric 
      ? [...value, Number(inputValue)] 
      : [...value, inputValue.trim()];
      
    onChange(newValue);
    setInputValue("");
    inputRef.current?.focus();
  };

  const handleRemoveItem = (index: number) => {
    const newValue = [...value];
    newValue.splice(index, 1);
    onChange(newValue);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddItem();
    }
  };

  const isValidInput = () => {
    if (!inputValue.trim()) return false;
    if (isNumeric) return !isNaN(Number(inputValue));
    return true;
  };

  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <Input
          ref={inputRef}
          type={isNumeric ? "number" : "text"}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="flex-1"
        />
        <Button 
          type="button" 
          size="icon"
          variant="outline"
          onClick={handleAddItem}
          disabled={!isValidInput()}
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>
      
      {value.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-2">
          {value.map((item, index) => (
            <Badge key={index} variant="secondary" className="px-2 py-1">
              {String(item)}
              <button
                type="button"
                className="ml-1 text-muted-foreground hover:text-foreground"
                onClick={() => handleRemoveItem(index)}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
} 