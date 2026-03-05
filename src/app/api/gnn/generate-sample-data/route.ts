import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  // Parse body once at the beginning
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { numSpots = 100, regionSize = 500 } = body;
  
  // Generate data locally (works on Vercel without external service)
  const proteins = ['EGFR', 'HER2', 'CD3', 'CD4', 'CD8', 'PD-1', 'PD-L1', 'Ki67', 'Vimentin', 'E-cadherin', 'CD20', 'CD56', 'FOXP3', 'CD68', 'MHC-II'];
  const cellTypes = ['T-cell', 'B-cell', 'NK-cell', 'Macrophage', 'Tumor', 'Fibroblast', 'Endothelial'];
  
  const spots = Array.from({ length: numSpots }, (_, i) => ({
    id: `spot-${i}`,
    x: Math.random() * regionSize,
    y: Math.random() * regionSize,
    protein: proteins[Math.floor(Math.random() * proteins.length)],
    expression: Math.random() * 100,
    cellType: cellTypes[Math.floor(Math.random() * cellTypes.length)]
  }));
  
  return NextResponse.json({ success: true, data: spots });
}
