
import React from "react";
import { toast } from "sonner";
import { XCircle } from "lucide-react";

interface NotificationProps {
  message: string;
}

export const showNotification = ({ message }: NotificationProps) => {
  toast(message, {
    icon: <XCircle className="text-blue-400" />,
    duration: 3000,
    className: "glass-dark text-white border border-white/20"
  });
};
