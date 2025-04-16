
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Shield, Copy, Check, Code } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import KyaroLogo from "./KyaroLogo";

interface KyaroBadgeProps {
  size?: "sm" | "md" | "lg";
}

const KyaroBadge: React.FC<KyaroBadgeProps> = ({ size = "md" }) => {
  const [copied, setCopied] = useState(false);
  
  const sizesMap = {
    sm: { badge: "h-6 text-xs", logo: 16 },
    md: { badge: "h-8 text-sm", logo: 20 },
    lg: { badge: "h-10 text-base", logo: 24 }
  };
  
  const badgeCode = `<a href="https://kyaro.com" target="_blank" rel="noopener noreferrer" class="inline-flex items-center px-3 py-1 space-x-2 rounded-full bg-[#1A1F2C] text-white hover:bg-opacity-80 transition-all">
  <!-- Replace SVG with your logo SVG -->
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" fill="#9B87F5"/>
    <path d="M15 9L9 15M9 9L15 15" stroke="white" stroke-width="2" stroke-linecap="round"/>
  </svg>
  <span>Bảo vệ bởi Kyaro Anti-DDoS</span>
</a>`;

  const copyCode = () => {
    navigator.clipboard.writeText(badgeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <div className={`inline-flex items-center px-3 ${sizesMap[size].badge} space-x-2 rounded-full bg-[#1A1F2C] text-white hover:bg-opacity-80 transition-all`}>
        <KyaroLogo size={sizesMap[size].logo} />
        <span>Bảo vệ bởi Kyaro Anti-DDoS</span>
      </div>

      <Dialog>
        <DialogTrigger asChild>
          <Button variant="ghost" size="sm" className="ml-2">
            <Code size={16} />
            <span className="ml-1">Get Badge</span>
          </Button>
        </DialogTrigger>
        <DialogContent className="glass-dark text-white border border-white/20">
          <DialogHeader>
            <DialogTitle>Kyaro Anti-DDoS Badge</DialogTitle>
            <DialogDescription className="text-white/70">
              Copy this code to your website to show the Kyaro Anti-DDoS badge.
            </DialogDescription>
          </DialogHeader>
          
          <div className="bg-black/30 p-4 rounded-md overflow-x-auto font-mono text-sm">
            {badgeCode}
          </div>
          
          <DialogFooter>
            <Button 
              onClick={copyCode} 
              className="flex items-center space-x-2"
            >
              {copied ? <Check size={16} /> : <Copy size={16} />}
              <span>{copied ? "Copied!" : "Copy code"}</span>
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default KyaroBadge;
