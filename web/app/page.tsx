"use client";

import { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Download, Trash2 } from "lucide-react";

interface GeneratedImage {
  id: string;
  prompt: string;
  status: "loading" | "complete" | "error";
  url?: string;
  error?: string;
  startTime: number;
  completionTime?: number;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes > 0 ? `${minutes}m ` : ''}${remainingSeconds}s`;
}

function Timer({ startTime }: { startTime: number }) {
  const [elapsed, setElapsed] = useState<number>(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Date.now() - startTime);
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime]);

  return <span>{formatDuration(elapsed)}</span>;
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [images, setImages] = useState<GeneratedImage[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('generated-images');
      return saved ? JSON.parse(saved) : [];
    }
    return [];
  });
  const [pendingJobs, setPendingJobs] = useState<string[]>([]);

  useEffect(() => {
    localStorage.setItem('generated-images', JSON.stringify(images));
  }, [images]);

  useEffect(() => {
    if (pendingJobs.length === 0) return;

    const intervalId = setInterval(async () => {
      const completedJobs: string[] = [];

      for (const jobId of pendingJobs) {
        if (!jobId || jobId === 'undefined') {
          completedJobs.push(jobId);
          continue;
        }

        try {
          const response = await fetch(`/api/results/${jobId}`);
          
          if (!response.ok) {
            throw new Error(`Failed to fetch results: ${response.statusText}`);
          }

          const data = await response.json();

          if (data.status === "SUCCESS") {
            setImages((prev) =>
              prev.map((img) =>
                img.id === jobId
                  ? { 
                      ...img, 
                      status: "complete", 
                      url: data.url,
                      completionTime: Date.now() 
                    }
                  : img
              )
            );
            completedJobs.push(jobId);
          } else if (data.status === "ERROR") {
            setImages((prev) =>
              prev.map((img) =>
                img.id === jobId
                  ? { 
                      ...img, 
                      status: "error", 
                      error: data.message,
                      completionTime: Date.now() 
                    }
                  : img
              )
            );
            completedJobs.push(jobId);
          }
        } catch (error) {
          console.error(`Error checking status for job ${jobId}:`, error);
        }
      }

      if (completedJobs.length > 0) {
        setPendingJobs((prev) =>
          prev.filter((jobId) => !completedJobs.includes(jobId))
        );
      }
    }, 1000);

    return () => clearInterval(intervalId);
  }, [pendingJobs]);

  const generateImage = async () => {
    if (!prompt.trim()) return;

    try {
      const formData = new FormData();
      formData.append("text", prompt);

      const response = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to generate image: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (!data["job-id"]) {
        throw new Error("No job ID received from server");
      }

      const newImage: GeneratedImage = {
        id: data["job-id"],
        prompt,
        status: "loading",
        startTime: Date.now(),
      };

      setImages((prev) => [newImage, ...prev]);
      setPendingJobs((prev) => [...prev, data["job-id"]]);
      setPrompt("");
    } catch (error) {
      console.error("Error generating image:", error);
      alert("Failed to generate image. Please try again.");
    }
  };

  const deleteImage = (id: string) => {
    setImages((prev) => prev.filter((img) => img.id !== id));
  };

  const downloadImage = async (url: string, prompt: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      
      const link = document.createElement('a');
      link.href = blobUrl;
      const filename = prompt.slice(0, 30).replace(/[^a-z0-9]/gi, '_').toLowerCase() + '.jpg';
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      console.error('Error downloading image:', error);
      alert('Failed to download image');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background via-background/95 to-background/90 font-mono relative overflow-hidden">
      <div className="absolute inset-0 bg-grid-white/[0.02] bg-grid-pattern animate-fade-in" />
      <div className="absolute inset-0 bg-gradient-to-b from-background via-transparent to-background pointer-events-none" />

      <header className="relative border-b border-border/20 bg-background/60 backdrop-blur-xl">
        <div className="container mx-auto px-6 py-8">
          <div className="space-y-1 animate-slide-down">
            <h1 className="text-5xl font-bold tracking-tight">
            TSAI EMLO-V4
            </h1>
            <p className="text-lg text-muted-foreground">
              Stable Diffusion 3
            </p>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-16 relative">
        <div className="max-w-4xl mx-auto mb-24 animate-slide-up">
          <div className="flex gap-4">
            <Input
              placeholder="Describe what you want to see..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="flex-1 h-16 text-xl font-mono bg-background/50 border-border/20 focus-visible:ring-offset-0 rounded-2xl px-6"
              onKeyDown={(e) => {
                if (e.key === "Enter") generateImage();
              }}
            />
            <Button 
              onClick={generateImage} 
              className="h-16 px-12 text-xl bg-foreground hover:bg-foreground/90 font-mono rounded-2xl transition-all hover:scale-105"
            >
              Generate
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {images.map((image, index) => (
            <div
              key={image.id}
              className="animate-slide-up"
              style={{ 
                animationDelay: `${index * 100}ms`,
                opacity: 0,
                animation: 'slide-up 0.5s ease forwards'
              }}
            >
              <Card className="overflow-hidden border-border/20 bg-background/50 backdrop-blur-xl hover:shadow-xl transition-all hover:-translate-y-1">
                <CardContent className="p-6">
                  <div className="aspect-square relative bg-muted/50 rounded-xl overflow-hidden group">
                    {image.status === "loading" ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/60 backdrop-blur-sm">
                        <Loader2 className="h-10 w-10 animate-spin mb-4" />
                        <div className="text-lg">
                          <Timer startTime={image.startTime} />
                        </div>
                      </div>
                    ) : image.status === "error" ? (
                      <div className="absolute inset-0 flex items-center justify-center text-destructive bg-background/60 backdrop-blur-sm">
                        Error: {image.error}
                      </div>
                    ) : (
                      <>
                        <img
                          src={image.url}
                          alt={image.prompt}
                          className="w-full h-full object-cover transition-transform duration-300"
                        />
                        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-4">
                          <Button
                            onClick={() => downloadImage(image.url!, image.prompt)}
                            variant="ghost"
                            size="icon"
                            className="h-10 w-10 rounded-full bg-background/20 hover:bg-background/40"
                          >
                            <Download className="h-5 w-5" />
                          </Button>
                          <Button
                            onClick={() => deleteImage(image.id)}
                            variant="ghost"
                            size="icon"
                            className="h-10 w-10 rounded-full bg-background/20 hover:bg-background/40 hover:text-destructive"
                          >
                            <Trash2 className="h-5 w-5" />
                          </Button>
                        </div>
                      </>
                    )}
                  </div>
                  <div className="mt-6 space-y-3">
                    <p className="text-base text-foreground/90 line-clamp-2 hover:line-clamp-none">
                      {image.prompt}
                    </p>
                    {image.completionTime && (
                      <p className="text-sm text-muted-foreground">
                        Generated in {formatDuration(image.completionTime - image.startTime)}
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>

        {images.length === 0 && (
          <div 
            className="text-center text-muted-foreground mt-20 animate-fade-in"
          >
            <p className="text-xl">No images generated yet. Enter a prompt to get started!</p>
          </div>
        )}
      </main>
    </div>
  );
}
