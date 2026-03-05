import { NextRequest, NextResponse } from 'next/server';
import { randomUUID } from 'crypto';

export async function POST(request: NextRequest) {
  // Parse body once at the beginning
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { spots, threshold = 50 } = body;
  
  if (!spots || !Array.isArray(spots) || spots.length === 0) {
    return NextResponse.json({ error: 'No spots provided' }, { status: 400 });
  }
  
  // Build graph locally (works on Vercel without external service)
  const nodes = spots.map((spot: { id: string; x: number; y: number; cellType?: string }) => ({
    id: spot.id,
    position: { x: spot.x, y: spot.y },
    label: spot.cellType
  }));
  
  const edges: { source: string; target: string; weight: number }[] = [];
  
  for (let i = 0; i < spots.length; i++) {
    for (let j = i + 1; j < spots.length; j++) {
      const dx = spots[i].x - spots[j].x;
      const dy = spots[i].y - spots[j].y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      
      if (distance < threshold) {
        edges.push({
          source: spots[i].id,
          target: spots[j].id,
          weight: 1 / (1 + distance / 10)
        });
      }
    }
  }
  
  const sessionId = `session-${Date.now()}-${randomUUID()}`;
  
  return NextResponse.json({
    success: true,
    sessionId,
    graph: {
      nodes,
      edges,
      stats: {
        nodeCount: nodes.length,
        edgeCount: edges.length,
        avgDegree: nodes.length > 0 ? (edges.length * 2) / nodes.length : 0
      }
    }
  });
}
