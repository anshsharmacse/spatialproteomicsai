import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  // Parse body once at the beginning
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 });
  }

  const { sessionId } = body;
  
  if (!sessionId) {
    return NextResponse.json({ error: 'Session ID is required' }, { status: 400 });
  }
  
  // Simulate realistic training progression (works on Vercel)
  const history = [];
  const epochs = 50;
  
  // Start with ~28% accuracy, progress to 82%
  const startAccuracy = 0.28;
  const targetAccuracy = 0.82;
  const startLoss = 1.8;
  const targetLoss = 0.35;
  
  for (let epoch = 0; epoch < epochs; epoch++) {
    // Calculate progress (0 to 1)
    const progress = epoch / (epochs - 1);
    
    // Use sigmoid-like curve for smooth progression
    const sigmoidProgress = 1 / (1 + Math.exp(-10 * (progress - 0.5)));
    
    // Calculate accuracy with some noise
    const baseAccuracy = startAccuracy + (targetAccuracy - startAccuracy) * sigmoidProgress;
    const noise = (Math.random() - 0.5) * 0.08 * (1 - sigmoidProgress);
    const accuracy = Math.max(0.2, Math.min(0.95, baseAccuracy + noise));
    
    // Calculate loss (inverse relationship with accuracy)
    const baseLoss = startLoss - (startLoss - targetLoss) * sigmoidProgress;
    const lossNoise = (Math.random() - 0.5) * 0.15 * (1 - sigmoidProgress);
    const loss = Math.max(0.2, baseLoss + lossNoise);
    
    history.push({ 
      epoch, 
      loss: Math.round(loss * 1000) / 1000, 
      accuracy: Math.round(accuracy * 1000) / 1000 
    });
  }
  
  return NextResponse.json({
    success: true,
    history,
    finalAccuracy: history[history.length - 1].accuracy
  });
}
