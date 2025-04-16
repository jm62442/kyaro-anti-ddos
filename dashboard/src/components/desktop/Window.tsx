
import React, { useState, useRef, useEffect } from 'react';
import { Rnd } from 'react-rnd';
import { X, Minus, Maximize, Minimize } from 'lucide-react';

interface WindowProps {
  id: string;
  title: string;
  icon: React.ReactNode;
  isActive: boolean;
  defaultPosition?: { x: number; y: number };
  defaultSize?: { width: number | string; height: number | string };
  onClose: () => void;
  onFocus: () => void;
  onMinimize: () => void; // Add new prop for minimization
  children: React.ReactNode;
}

const Window: React.FC<WindowProps> = ({
  id,
  title,
  icon,
  isActive,
  defaultPosition = { x: 100, y: 100 },
  defaultSize = { width: 800, height: 500 },
  onClose,
  onFocus,
  onMinimize, // Add minimization handler
  children,
}) => {
  const [isAnimatingClose, setIsAnimatingClose] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const windowRef = useRef<Rnd | null>(null);
  const [prevSize, setPrevSize] = useState(defaultSize);
  const [prevPosition, setPrevPosition] = useState(defaultPosition);
  const [taskbarHeight, setTaskbarHeight] = useState(48);
  const desktopRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Find the desktop and taskbar elements for positioning
    desktopRef.current = document.querySelector('.desktop-container');
    const taskbar = document.querySelector('.taskbar');
    if (taskbar) {
      setTaskbarHeight(taskbar.clientHeight);
    }

    if (isAnimatingClose) {
      const timer = setTimeout(() => {
        onClose();
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isAnimatingClose, onClose]);

  const handleClose = () => {
    setIsAnimatingClose(true);
  };

  const handleMaximize = () => {
    if (!isMaximized) {
      if (windowRef.current) {
        const { width, height } = windowRef.current.getSelfElement().getBoundingClientRect();
        const { x, y } = windowRef.current.getDraggablePosition();
        setPrevSize({ width, height });
        setPrevPosition({ x, y });
      }
      setIsMaximized(true);
    } else {
      setIsMaximized(false);
      if (windowRef.current) {
        // Use a timeout to ensure state has updated before repositioning
        setTimeout(() => {
          windowRef.current?.updatePosition(prevPosition);
          windowRef.current?.updateSize({ width: prevSize.width, height: prevSize.height });
        }, 50);
      }
    }
  };

  const handleMinimize = () => {
    onMinimize(); // Call the minimization handler
  };

  // Calculate maximized dimensions
  const getMaximizedDimensions = () => {
    if (desktopRef.current) {
      const desktopRect = desktopRef.current.getBoundingClientRect();
      return {
        width: desktopRect.width,
        height: desktopRect.height - taskbarHeight,
        x: 0, 
        y: 0
      };
    }
    // Fallback to window dimensions if desktop element not found
    return {
      width: window.innerWidth,
      height: window.innerHeight - taskbarHeight,
      x: 0,
      y: 0
    };
  };

  const maxDimensions = getMaximizedDimensions();

  return (
    <Rnd
      ref={windowRef}
      default={{
        ...defaultPosition,
        ...defaultSize,
      }}
      style={{
        zIndex: isActive ? 100 : 10,
        opacity: isAnimatingClose ? 0 : 1,
        transform: isAnimatingClose ? 'scale(0.95)' : 'scale(1)',
        transition: 'opacity 0.3s ease-out, transform 0.3s ease-out',
      }}
      className={`window rounded-lg overflow-hidden bg-[hsl(var(--card))] animate-window-open ${isActive ? 'ring-2 ring-blue-500/50' : ''}`}
      dragHandleClassName="window-drag-handle"
      bounds="parent"
      minWidth={300}
      minHeight={200}
      size={isMaximized ? { width: maxDimensions.width, height: maxDimensions.height } : undefined}
      position={isMaximized ? { x: maxDimensions.x, y: maxDimensions.y } : undefined}
      disableDragging={isMaximized}
      enableResizing={!isMaximized}
      onDragStart={onFocus}
      onResizeStart={onFocus}
      onMouseDown={onFocus}
    >
      <div 
        className="window-header window-drag-handle cursor-move flex items-center justify-between bg-[hsl(var(--card-header))] px-4 py-2"
      >
        <div className="flex items-center space-x-2">
          {icon}
          <span className="text-white font-medium">{title}</span>
        </div>
        <div className="window-controls flex space-x-2">
          <button 
            className="control-button bg-gray-400 hover:bg-gray-300 p-1.5 rounded"
            onClick={handleMinimize}
          >
            <Minus size={10} className="text-gray-700" />
          </button>
          <button 
            className="control-button bg-blue-400 hover:bg-blue-300 p-1.5 rounded"
            onClick={handleMaximize}
          >
            <Maximize size={10} className="text-blue-700" />
          </button>
          <button 
            className="control-button bg-red-400 hover:bg-red-300 p-1.5 rounded"
            onClick={handleClose}
          >
            <X size={10} className="text-red-700" />
          </button>
        </div>
      </div>
      <div 
        className="p-4 h-[calc(100%-40px)] overflow-auto"
        onClick={onFocus}
      >
        {children}
      </div>
    </Rnd>
  );
};

export default Window;
