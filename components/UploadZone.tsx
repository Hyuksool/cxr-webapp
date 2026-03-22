"use client";

import { useCallback, useState } from "react";

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function UploadZone({ onFileSelect, disabled }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        alert("Please upload an image file (JPEG, PNG, etc.)");
        return;
      }
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile, disabled]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors ${
        isDragging
          ? "border-blue-500 bg-blue-50"
          : disabled
          ? "border-gray-200 bg-gray-50 opacity-60"
          : "border-gray-300 bg-white hover:border-blue-400 hover:bg-blue-50"
      }`}
      onDrop={handleDrop}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
    >
      <div className="flex flex-col items-center gap-3">
        {/* X-Ray icon */}
        <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
          <svg
            className="w-8 h-8 text-blue-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
        </div>
        <div>
          <p className="text-sm font-medium text-gray-700">
            Drop chest X-ray image here, or{" "}
            <label
              className={`text-blue-600 underline ${
                disabled ? "cursor-not-allowed" : "cursor-pointer hover:text-blue-800"
              }`}
            >
              browse
              <input
                type="file"
                className="hidden"
                accept="image/*"
                disabled={disabled}
                onChange={handleInputChange}
              />
            </label>
          </p>
          <p className="text-xs text-gray-400 mt-1">
            JPEG, PNG, DICOM-preview — max 20MB
          </p>
        </div>
      </div>
    </div>
  );
}
