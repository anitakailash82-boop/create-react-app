/**
 * DakshTutor Backend - server.js
 * - Express server with endpoints: /api/chat, /api/notes, /api/mcqs
 * - Firebase ID token verification middleware (optional)
 * - Rate limiting + CORS + Helmet + logging
 *
 * Node >= 18 recommended. Uses ES modules.
 */

import express from "express";
import dotenv from "dotenv";
import helmet from "helmet";
import cors from "cors";
import morgan from "morgan";
import fetch from "node-fetch";
import rateLimit from "express-rate-limit";
import Joi from "joi";
import fs from "fs";
import path from "path";
import admin from "firebase-admin";

dotenv.config();

const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("ERROR: OPENAI_API_KEY not set in environment.");
  process.exit(1);
}

/** ---------- Optional: initialize Firebase Admin for token verification ---------- */
let firebaseEnabled = false;
if (process.env.FIREBASE_PROJECT_ID && process.env.FIREBASE_PRIVATE_KEY && process.env.FIREBASE_CLIENT_EMAIL) {
  try {
    const privateKey = process.env.FIREBASE_PRIVATE_KEY.replace(/\\n/g, "\n");
    admin.initializeApp({
      credential: admin.credential.cert({
        projectId: process.env.FIREBASE_PROJECT_ID,
        clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
        privateKey,
      }),
    });
    firebaseEnabled = true;
    console.log("Firebase Admin initialized.");
  } catch (err) {
    console.warn("Firebase Admin init failed:", err.message);
  }
}

/** ---------- Express setup ---------- */
const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "120kb" })); // limit payload size
app.use(morgan("combined"));

/** ---------- Rate limiting ---------- */
const windowMinutes = parseInt(process.env.RATE_LIMIT_WINDOW_MINUTES || "1", 10);
const windowMs = windowMinutes * 60 * 1000;
const maxRequests = parseInt(process.env.RATE_LIMIT_MAX_REQUESTS_PER_WINDOW || "30", 10);

const globalLimiter = rateLimit({
  windowMs,
  max: maxRequests,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests, please try again later." },
});
app.use(globalLimiter);

/** ---------- Helper: call OpenAI Chat Completions ---------- */
const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";

// Basic wrapper
async function callOpenAIChat(messages = [], model = "gpt-4o-mini", max_tokens = 900, temperature = 0.2) {
  const resp = await fetch(OPENAI_CHAT_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      messages,
      max_tokens,
      temperature,
    }),
  });

  if (!resp.ok) {
    const text = await resp.text();
    const err = new Error(`OpenAI API error: ${resp.status} ${resp.statusText} - ${text}`);
    err.status = resp.status;
    throw err;
  }
  const data = await resp.json();
  return data;
}

/** ---------- Middleware: optional Firebase token verification ---------- */
async function firebaseAuthMiddleware(req, res, next) {
  if (!firebaseEnabled) return next();
  const authHeader = req.headers.authorization || "";
  const match = authHeader.match(/^Bearer (.+)$/);
  if (!match) {
    return res.status(401).json({ error: "Missing or invalid Authorization header (expected Bearer <Firebase ID token>)" });
  }
  const idToken = match[1];
  try {
    const decoded = await admin.auth().verifyIdToken(idToken);
    req.user = { uid: decoded.uid, email: decoded.email || null };
    return next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired Firebase ID token", detail: err.message });
  }
}

/** ---------- Validation schemas ---------- */
const chatSchema = Joi.object({
  messages: Joi.array().items(
    Joi.object({ role: Joi.string().valid("system", "user", "assistant").required(), content: Joi.string().required() })
  ).min(1).required(),
  model: Joi.string().optional(),
  max_tokens: Joi.number().integer().min(50).max(4000).optional(),
  temperature: Joi.number().min(0).max(1).optional()
});

const notesSchema = Joi.object({
  topic: Joi.string().min(3).max(200).required(),
  board: Joi.string().optional().default("RBSE"),
  class: Joi.number().integer().min(1).max(12).optional().default(12)
});

const mcqSchema = Joi.object({
  topic: Joi.string().min(3).max(200).required(),
  num: Joi.number().integer().min(1).max(50).optional().default(10),
  difficulty: Joi.string().valid("easy","medium","hard").optional().default("medium")
});

/** ---------- Routes ---------- */

/**
 * POST /api/chat
 * Body: { messages: [{role, content}], model?, max_tokens?, temperature? }
 * Returns raw OpenAI response.
 */
app.post("/api/chat", firebaseAuthMiddleware, async (req, res) => {
  // validate
  const { error, value } = chatSchema.validate(req.body);
  if (error) return res.status(400).json({ error: error.details.map(d => d.message).join(", ") });

  const { messages, model = "gpt-4o-mini", max_tokens = 900, temperature = 0.2 } = value;

  // Add server-side system prompt enforcing RBSE commerce style if not present
  const hasSystem = messages.some(m => m.role === "system");
  const systemMessage = {
    role: "system",
    content:
      'You are "RBSE Commerce Tutor", an experienced teacher for RBSE Class 12 Commerce. Keep language simple, provide stepwise solutions when asked, produce concise notes when asked, and for MCQs return 4 options and a short explanation. Be succinct and exam-aligned.'
  };

  const finalMessages = hasSystem ? messages : [systemMessage, ...messages];

  try {
    const data = await callOpenAIChat(finalMessages, model, max_tokens, temperature);
    // Return the OpenAI response as-is (client will parse)
    res.json(data);
  } catch (err) {
    console.error("OpenAI chat error:", err.message);
    res.status(err.status || 500).json({ error: "OpenAI chat failed", detail: err.message });
  }
});


/**
 * POST /api/notes
 * Body: { topic, board, class }
 * Returns: { notes: "..." }
 */
app.post("/api/notes", firebaseAuthMiddleware, async (req, res) => {
  const { error, value } = notesSchema.validate(req.body);
  if (error) return res.status(400).json({ error: error.details.map(d => d.message).join(", ") });

  const { topic, board, class: cls } = value;

  const messages = [
    {
      role: "system",
      content: 'You are "RBSE Commerce Tutor". Produce concise study notes for RBSE Class 12 Commerce. Use bullet points, include definitions, key points, important formulas (if any), and one solved example. Keep output short and printable on one A4 page.'
    },
    {
      role: "user",
      content: `Generate concise notes for "${topic}" for ${board} Class ${cls}. Output plain text (not JSON).`
    }
  ];

  try {
    const data = await callOpenAIChat(messages, "gpt-4o-mini", 700, 0.2);
    const reply = data.choices?.[0]?.message?.content ?? "";
    res.json({ notes: reply });
  } catch (err) {
    console.error("Notes generation failed:", err.message);
    res.status(500).json({ error: "notes generation failed", detail: err.message });
  }
});


/**
 * POST /api/mcqs
 * Body: { topic, num, difficulty }
 * Returns: { mcqs: [ { question, options, answerIndex, explanation } ] }
 *
 * The endpoint expects the model to return a valid JSON array. We attempt to parse defensively.
 */
app.post("/api/mcqs", firebaseAuthMiddleware, async (req, res) => {
  const { error, value } = mcqSchema.validate(req.body);
  if (error) return res.status(400).json({ error: error.details.map(d => d.message).join(", ") });

  const { topic, num, difficulty } = value;

  const messages = [
    {
      role: "system",
      content:
        'You are "RBSE Class 12 Commerce MCQ generator". Return ONLY valid JSON: an array of objects. Each object must be { "question": "...", "options": ["a","b","c","d"], "answerIndex": <0-3>, "explanation": "..." }. No extra text.'
    },
    {
      role: "user",
      content: `Create ${num} multiple-choice questions for the chapter "${topic}" (difficulty: ${difficulty}). Return a single JSON array only.`
    }
  ];

  try {
    const data = await callOpenAIChat(messages, "gpt-4o-mini", 1200, 0.1);
    const raw = data.choices?.[0]?.message?.content ?? "";

    // Defensive JSON extraction
    let parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      // Try to locate first '[' and last ']' and parse that slice
      const start = raw.indexOf("[");
      const end = raw.lastIndexOf("]");
      if (start >= 0 && end > start) {
        const slice = raw.slice(start, end + 1);
        parsed = JSON.parse(slice);
      } else {
        throw new Error("Could not parse JSON from model output");
      }
    }

    // Validate shape of parsed value
    if (!Array.isArray(parsed)) throw new Error("Parsed MCQ output is not an array");

    // Validate each item shape conservatively
    const normalized = parsed.map((item, i) => {
      // ensure fields exist
      const question = (item.question || "").toString();
      const options = Array.isArray(item.options) ? item.options.map(String) : [];
      const answerIndex = Number(item.answerIndex);
      const explanation = (item.explanation || "").toString();

      if (options.length !== 4 || Number.isNaN(answerIndex) || answerIndex < 0 || answerIndex > 3) {
        throw new Error(`MCQ at index ${i} has invalid structure`);
      }
      return { question, options, answerIndex, explanation };
    });

    res.json({ mcqs: normalized });
  } catch (err) {
    console.error("MCQ generation error:", err.message);
    res.status(500).json({ error: "mcq generation failed", detail: err.message });
  }
});

/** ---------- Health & fallback ---------- */
app.get("/healthz", (req, res) => res.json({ status: "ok" }));

/** ---------- Global error handler ---------- */
app.use((err, req, res, next) => {
  console.error("Unhandled error:", err);
  res.status(500).json({ error: "internal_server_error", detail: err?.message || "unknown" });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
