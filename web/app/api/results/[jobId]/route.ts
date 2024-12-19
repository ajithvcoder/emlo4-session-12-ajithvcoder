import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:9080";

type Props = {
  params: Promise<{ jobId: string }>
}

export async function GET(request: NextRequest, { params }: Props) {
  try {
    // Await the params object to get the jobId
    const { jobId } = await params;
    
    // Validate the jobId parameter
    if (typeof jobId !== 'string') {
      return NextResponse.json(
        { error: "Invalid job ID format" },
        { status: 400 }
      );
    }

    // Create URL object to ensure proper URL construction
    const url = new URL(`${BACKEND_URL}/results/${jobId}`);
    
    const response = await fetch(url.toString());
    
    if (!response.ok) {
      throw new Error(`Backend responded with status ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching results:", error);
    return NextResponse.json(
      { error: "Failed to fetch results" },
      { status: 500 }
    );
  }
} 