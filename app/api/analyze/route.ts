import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_GEMINI_API_KEY!);

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();
    
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-3.8-flash',
      generationConfig: {
        temperature: 0.7,
        maxOutputTokens: 80,
      }
    });

    const result = await model.generateContent(prompt);
    const text = result.response.text().trim();
    
    return NextResponse.json({ correction: text });
  } catch (error) {
    console.error('Gemini API error:', error);
    return NextResponse.json(
      { error: 'Failed to generate correction' },
      { status: 500 }
    );
  }
}
