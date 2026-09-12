import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { image } = await req.json();

    if (!image) {
      return NextResponse.json({ error: 'No image provided' }, { status: 400 });
    }

    const heavyModelUrl =
      process.env.HEAVY_MODEL_URL ||
      process.env.NEXT_PUBLIC_HEAVY_MODEL_URL ||
      'http://127.0.0.1:8000';

    const targetEndpoint = `${heavyModelUrl.replace(/\/$/, '')}/predict`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000); // 6s timeout

    try {
      const response = await fetch(targetEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        return NextResponse.json(
          { error: `Heavy model server responded with status ${response.status}: ${errorText}` },
          { status: response.status }
        );
      }

      const data = await response.json();
      return NextResponse.json(data);
    } catch (fetchErr: any) {
      clearTimeout(timeoutId);
      if (fetchErr.name === 'AbortError') {
        return NextResponse.json(
          { error: 'Heavy model server timed out after 6 seconds' },
          { status: 504 }
        );
      }
      return NextResponse.json(
        {
          error: `Failed to connect to heavy model server at ${heavyModelUrl}. Ensure the Python server and Cloudflare tunnel are running.`,
          detail: fetchErr.message,
        },
        { status: 502 }
      );
    }
  } catch (err: any) {
    return NextResponse.json(
      { error: err.message || 'Internal Server Error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  const heavyModelUrl =
    process.env.HEAVY_MODEL_URL ||
    process.env.NEXT_PUBLIC_HEAVY_MODEL_URL ||
    'http://127.0.0.1:8000';

  try {
    const response = await fetch(`${heavyModelUrl.replace(/\/$/, '')}/health`, {
      cache: 'no-store',
    });
    if (response.ok) {
      const data = await response.json();
      return NextResponse.json({ online: true, url: heavyModelUrl, ...data });
    }
    return NextResponse.json({ online: false, url: heavyModelUrl, status: response.status });
  } catch (err: any) {
    return NextResponse.json({ online: false, url: heavyModelUrl, error: err.message });
  }
}
