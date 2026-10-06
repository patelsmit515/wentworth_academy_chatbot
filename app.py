"""
Wentworth Academy NLP Chatbot
Custom web interface for the existing NLP backend.
"""

import json
import os
import sys
import threading
import webbrowser

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from src.data_loader import load_intents, remove_unknown
from src.features import add_text_features
from src.prediction import predict_intent
from src.response import get_response
from src.training_mlp import train_mlp


# ============================================================
# Configuration
# ============================================================

HOST = "127.0.0.1"
PORT = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "intents.csv")


# ============================================================
# Global model objects
# ============================================================

MODEL = None
PREPROCESSOR = None

DATA_STATS = {
    "examples": 0,
    "intents": 0,
}


# ============================================================
# Approved example questions
# ============================================================

APPROVED_EXAMPLES = [
    "Hi there",
    "I need help with algebra",
    "Who teaches chemistry?",
    "Who teaches Mathematics?",
    "Who teaches Biology?",
    "Who teaches Physics?",
    "Who teaches English?",
    "Who teaches History?",
    "Where can I find the library?",
    "Where is the Science Building?",
    "Where are the laboratories?",
    "Where is the cafeteria?",
    "Where is the Student Center?",
    "Where is the gym?",
    "What clubs can I join?",
    "What clubs are available?",
    "How can I join a club?",
    "What does the Debate Club do?",
    "When are the final exams?",
    "I don't understand physics.",
    "Goodbye",
]


# ============================================================
# HTML / CSS / JavaScript
# ============================================================

HTML_CONTENT = r"""
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Wentworth Academy</title>

    <style>

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        :root {
            --navy-950: #07111f;
            --navy-900: #0b1728;
            --navy-850: #101f34;
            --navy-800: #14263e;
            --navy-700: #1b3150;

            --gold: #d7b56d;
            --gold-light: #ead49b;

            --text-main: #edf2f8;
            --text-soft: #aab7c7;
            --text-muted: #75849a;

            --border: rgba(255, 255, 255, 0.09);
            --border-gold: rgba(215, 181, 109, 0.35);

            --user-message: #1d3656;
            --assistant-message: #14243a;

            --green: #59c58b;

            --shadow: 0 20px 50px rgba(0, 0, 0, 0.28);
        }

        body {
            font-family:
                Inter,
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            background:
                radial-gradient(
                    circle at top right,
                    rgba(36, 65, 100, 0.28),
                    transparent 35%
                ),
                var(--navy-950);

            color: var(--text-main);
            min-height: 100vh;
        }

        button,
        input {
            font-family: inherit;
        }

        button {
            cursor: pointer;
        }

        /* =====================================================
           Main Layout
           ===================================================== */

        .app {
            display: flex;
            min-height: 100vh;
        }

        /* =====================================================
           Sidebar
           ===================================================== */

        .sidebar {
            width: 300px;
            flex-shrink: 0;

            background:
                linear-gradient(
                    180deg,
                    #0d1b2d 0%,
                    #091523 100%
                );

            border-right: 1px solid var(--border);

            display: flex;
            flex-direction: column;

            overflow-y: auto;
        }

        .sidebar-header {
            padding: 26px 22px 20px;

            border-bottom: 1px solid var(--border);
        }

        .academy-name {
            font-size: 19px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        .academy-subtitle {
            margin-top: 5px;
            color: var(--text-muted);
            font-size: 12px;
        }

        .sidebar-content {
            padding: 18px 16px;
        }

        .section-title {
            color: var(--gold);
            font-size: 10px;
            font-weight: 700;

            text-transform: uppercase;
            letter-spacing: 1.5px;

            margin: 20px 8px 9px;
        }

        .section-title:first-child {
            margin-top: 0;
        }

        .category-item {
            display: flex;
            align-items: center;

            min-height: 36px;
            padding: 7px 9px;

            border-radius: 7px;

            color: var(--text-soft);
            font-size: 13px;

            transition:
                background 0.15s ease,
                color 0.15s ease;
        }

        .category-item:hover {
            background: rgba(255, 255, 255, 0.045);
            color: var(--text-main);
        }

        /* =====================================================
           Examples
           ===================================================== */

        .examples {
            margin-top: 7px;
        }

        .example-button {
            display: block;
            width: 100%;

            text-align: left;

            padding: 9px 10px;
            margin-bottom: 5px;

            background: rgba(255, 255, 255, 0.025);

            border: 1px solid transparent;
            border-radius: 7px;

            color: #b9c5d3;
            font-size: 12px;
            line-height: 1.35;

            transition:
                background 0.15s ease,
                border 0.15s ease,
                color 0.15s ease;
        }

        .example-button:hover {
            background: rgba(215, 181, 109, 0.08);
            border-color: var(--border-gold);
            color: var(--gold-light);
        }

        /* =====================================================
           Sidebar stats
           ===================================================== */

        .stats {
            margin-top: 22px;

            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 7px;
        }

        .stat-card {
            padding: 11px;

            background: rgba(255, 255, 255, 0.035);

            border: 1px solid var(--border);
            border-radius: 8px;
        }

        .stat-value {
            font-size: 17px;
            font-weight: 700;
            color: var(--gold-light);
        }

        .stat-label {
            margin-top: 3px;

            color: var(--text-muted);
            font-size: 10px;
        }

        /* =====================================================
           Main Area
           ===================================================== */

        .main {
            flex: 1;
            min-width: 0;

            display: flex;
            flex-direction: column;

            min-height: 100vh;
        }

        /* =====================================================
           Header
           ===================================================== */

        .topbar {
            min-height: 76px;

            display: flex;
            align-items: center;
            justify-content: space-between;

            padding: 0 28px;

            background:
                linear-gradient(
                    135deg,
                    #10233b,
                    #0c1b2e
                );

            border-bottom: 1px solid var(--border);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-mark {
            width: 38px;
            height: 38px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 8px;

            background:
                linear-gradient(
                    145deg,
                    #203b5c,
                    #13263f
                );

            border: 1px solid var(--border-gold);

            color: var(--gold-light);

            font-size: 13px;
            font-weight: 800;
            letter-spacing: 1px;
        }

        .brand-title {
            font-size: 16px;
            font-weight: 700;
        }

        .brand-subtitle {
            color: var(--text-muted);
            font-size: 11px;
            margin-top: 2px;
        }

        .status-area {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 7px;

            color: #aebccc;
            font-size: 12px;
        }

        .status-dot {
            width: 7px;
            height: 7px;

            border-radius: 50%;
            background: var(--green);

            box-shadow:
                0 0 0 4px rgba(89, 197, 139, 0.08);
        }

        .model-badge {
            padding: 6px 9px;

            border: 1px solid var(--border);
            border-radius: 6px;

            color: var(--text-muted);
            font-size: 10px;
        }

        /* =====================================================
           Chat
           ===================================================== */

        .chat-wrapper {
            flex: 1;

            width: 100%;
            max-width: 1050px;

            margin: 0 auto;

            display: flex;
            flex-direction: column;

            min-height: 0;
        }

        .chat-toolbar {
            display: flex;
            justify-content: flex-end;

            padding: 15px 28px 0;
        }

        .clear-button {
            background: transparent;

            border: 1px solid var(--border);
            border-radius: 6px;

            color: var(--text-muted);

            padding: 7px 11px;

            font-size: 11px;

            transition:
                color 0.15s ease,
                border 0.15s ease;
        }

        .clear-button:hover {
            color: var(--text-main);
            border-color: rgba(255, 255, 255, 0.18);
        }

        .chat-area {
            flex: 1;

            overflow-y: auto;

            padding: 30px 28px 24px;
        }

        /* =====================================================
           Empty State
           ===================================================== */

        .empty-state {
            min-height: 52vh;

            display: flex;
            align-items: center;
            justify-content: center;

            text-align: center;
        }

        .empty-content {
            max-width: 570px;
        }

        .empty-mark {
            width: 58px;
            height: 58px;

            margin: 0 auto 20px;

            display: flex;
            align-items: center;
            justify-content: center;

            border-radius: 12px;

            background: rgba(215, 181, 109, 0.08);
            border: 1px solid var(--border-gold);

            color: var(--gold);

            font-size: 16px;
            font-weight: 800;
            letter-spacing: 1px;
        }

        .empty-title {
            font-size: 27px;
            font-weight: 700;

            letter-spacing: -0.5px;
        }

        .empty-description {
            margin-top: 11px;

            color: var(--text-muted);

            font-size: 14px;
            line-height: 1.7;
        }

        /* =====================================================
           Messages
           ===================================================== */

        .message {
            display: flex;

            margin-bottom: 20px;
        }

        .message.user {
            justify-content: flex-end;
        }

        .message.assistant {
            justify-content: flex-start;
        }

        .message-inner {
            max-width: min(760px, 82%);
        }

        .message-label {
            margin-bottom: 6px;

            color: var(--text-muted);

            font-size: 10px;
            font-weight: 700;

            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .user .message-label {
            text-align: right;
        }

        .message-bubble {
            padding: 13px 16px;

            border-radius: 11px;

            font-size: 14px;
            line-height: 1.6;

            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .user .message-bubble {
            background: var(--user-message);

            border: 1px solid rgba(101, 148, 197, 0.18);

            border-bottom-right-radius: 3px;
        }

        .assistant .message-bubble {
            background: var(--assistant-message);

            border: 1px solid var(--border);

            border-bottom-left-radius: 3px;
        }

        .message-meta {
            display: flex;
            align-items: center;
            gap: 9px;

            margin-top: 6px;

            color: var(--text-muted);

            font-size: 10px;
        }

        .user .message-meta {
            justify-content: flex-end;
        }

        .confidence {
            color: var(--gold);
        }

        /* =====================================================
           Typing indicator
           ===================================================== */

        .typing {
            display: flex;
            align-items: center;
            gap: 5px;

            padding: 4px 2px;
        }

        .typing span {
            width: 6px;
            height: 6px;

            border-radius: 50%;

            background: #78899d;

            animation: typing 1.2s infinite ease-in-out;
        }

        .typing span:nth-child(2) {
            animation-delay: 0.15s;
        }

        .typing span:nth-child(3) {
            animation-delay: 0.3s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.45;
            }

            30% {
                transform: translateY(-4px);
                opacity: 1;
            }
        }

        /* =====================================================
           Input
           ===================================================== */

        .input-section {
            padding: 0 28px 22px;
        }

        .input-container {
            display: flex;
            align-items: center;
            gap: 8px;

            padding: 7px;

            background: #101f32;

            border: 1px solid var(--border);

            border-radius: 10px;

            box-shadow: var(--shadow);
        }

        .input-container:focus-within {
            border-color: rgba(215, 181, 109, 0.42);
        }

        #message-input {
            flex: 1;

            min-width: 0;

            border: none;
            outline: none;

            background: transparent;

            color: var(--text-main);

            padding: 10px 12px;

            font-size: 14px;
        }

        #message-input::placeholder {
            color: #63748a;
        }

        .send-button {
            border: none;

            padding: 10px 17px;

            border-radius: 7px;

            background: var(--gold);
            color: #152033;

            font-size: 12px;
            font-weight: 700;

            transition:
                background 0.15s ease,
                transform 0.1s ease;
        }

        .send-button:hover {
            background: var(--gold-light);
        }

        .send-button:active {
            transform: translateY(1px);
        }

        .send-button:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }

        .disclaimer {
            margin-top: 8px;

            text-align: center;

            color: #53647a;

            font-size: 9px;
        }

        /* =====================================================
           Mobile
           ===================================================== */

        .mobile-menu {
            display: none;

            border: 1px solid var(--border);
            background: transparent;

            color: var(--text-soft);

            border-radius: 6px;

            width: 35px;
            height: 35px;

            font-size: 18px;
        }

        @media (max-width: 850px) {

            .sidebar {
                position: fixed;

                top: 0;
                bottom: 0;
                left: 0;

                z-index: 100;

                transform: translateX(-100%);

                transition: transform 0.25s ease;

                box-shadow: 20px 0 50px rgba(0, 0, 0, 0.35);
            }

            .sidebar.open {
                transform: translateX(0);
            }

            .mobile-menu {
                display: block;
            }

            .topbar {
                padding: 0 16px;
            }

            .brand {
                gap: 8px;
            }

            .brand-mark {
                display: none;
            }

            .model-badge {
                display: none;
            }

            .status-area {
                gap: 8px;
            }

            .chat-area {
                padding-left: 16px;
                padding-right: 16px;
            }

            .chat-toolbar {
                padding-left: 16px;
                padding-right: 16px;
            }

            .input-section {
                padding-left: 16px;
                padding-right: 16px;
            }

            .message-inner {
                max-width: 90%;
            }

            .empty-title {
                font-size: 23px;
            }
        }

    </style>
</head>


<body>

<div class="app">

    <!-- ======================================================
         SIDEBAR
         ====================================================== -->

    <aside class="sidebar" id="sidebar">

        <div class="sidebar-header">

            <div class="academy-name">
                Wentworth Academy
            </div>

            <div class="academy-subtitle">
                Student Support Portal
            </div>

        </div>


        <div class="sidebar-content">

            <div class="section-title">
                Academics
            </div>

            <div class="category-item">
                Class schedules
            </div>

            <div class="category-item">
                Exam dates
            </div>

            <div class="category-item">
                Academic help
            </div>

            <div class="category-item">
                Teacher information
            </div>


            <div class="section-title">
                Campus
            </div>

            <div class="category-item">
                Library
            </div>

            <div class="category-item">
                Science Building
            </div>

            <div class="category-item">
                Laboratories
            </div>

            <div class="category-item">
                Cafeteria
            </div>

            <div class="category-item">
                Student Center
            </div>

            <div class="category-item">
                Gymnasium
            </div>

            <div class="category-item">
                Admin Office
            </div>


            <div class="section-title">
                Student Life
            </div>

            <div class="category-item">
                Science Club
            </div>

            <div class="category-item">
                Debate Club
            </div>

            <div class="category-item">
                Music Club
            </div>

            <div class="category-item">
                Art Club
            </div>

            <div class="category-item">
                Sports Clubs
            </div>

            <div class="category-item">
                School rules
            </div>

            <div class="category-item">
                Attendance
            </div>

            <div class="category-item">
                Uniform
            </div>


            <div class="section-title">
                General
            </div>

            <div class="category-item">
                Greetings
            </div>

            <div class="category-item">
                Goodbyes
            </div>

            <div class="category-item">
                Out-of-scope questions
            </div>


            <!-- =================================================
                 EXAMPLES
                 ================================================= -->

            <div class="section-title">
                Try an example
            </div>

            <div class="examples">

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Hi there">
                    Hi there
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="I need help with algebra">
                    I need help with algebra
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Who teaches chemistry?">
                    Who teaches chemistry?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Who teaches Mathematics?">
                    Who teaches Mathematics?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Who teaches Biology?">
                    Who teaches Biology?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Who teaches Physics?">
                    Who teaches Physics?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Who teaches English?">
                    Who teaches English?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Who teaches History?">
                    Who teaches History?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Where can I find the library?">
                    Where can I find the library?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Where is the Science Building?">
                    Where is the Science Building?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Where are the laboratories?">
                    Where are the laboratories?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Where is the cafeteria?">
                    Where is the cafeteria?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Where is the Student Center?">
                    Where is the Student Center?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Where is the gym?">
                    Where is the gym?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="What clubs can I join?">
                    What clubs can I join?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="What clubs are available?">
                    What clubs are available?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="How can I join a club?">
                    How can I join a club?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="What does the Debate Club do?">
                    What does the Debate Club do?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="When are the final exams?">
                    When are the final exams?
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="I don't understand physics.">
                    I don't understand physics.
                </button>

                <button
                    type="button"
                    class="example-button"
                    data-prompt="Goodbye">
                    Goodbye
                </button>

            </div>


            <!-- =================================================
                 PROJECT STATS
                 ================================================= -->

            <div class="section-title">
                Model
            </div>

            <div class="stats">

                <div class="stat-card">
                    <div class="stat-value">
                        3-Layer
                    </div>
                    <div class="stat-label">
                        MLP
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-value">
                        226
                    </div>
                    <div class="stat-label">
                        Examples
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-value">
                        94%
                    </div>
                    <div class="stat-label">
                        Accuracy
                    </div>
                </div>

                <div class="stat-card">
                    <div class="stat-value">
                        95%
                    </div>
                    <div class="stat-label">
                        Macro F1
                    </div>
                </div>

            </div>

        </div>

    </aside>


    <!-- ======================================================
         MAIN
         ====================================================== -->

    <main class="main">

        <header class="topbar">

            <div class="brand">

                <button
                    type="button"
                    class="mobile-menu"
                    id="mobile-menu">
                    ☰
                </button>

                <div class="brand-mark">
                    WA
                </div>

                <div>

                    <div class="brand-title">
                        Wentworth Academy
                    </div>

                    <div class="brand-subtitle">
                        Student Support Assistant
                    </div>

                </div>

            </div>


            <div class="status-area">

                <div class="status">

                    <span class="status-dot"></span>

                    Assistant Online

                </div>

                <div class="model-badge">
                    MLP Neural Network
                </div>

            </div>

        </header>


        <div class="chat-wrapper">

            <div class="chat-toolbar">

                <button
                    type="button"
                    class="clear-button"
                    id="clear-button">
                    Clear conversation
                </button>

            </div>


            <div class="chat-area" id="chat-area">

                <div class="empty-state" id="empty-state">

                    <div class="empty-content">

                        <div class="empty-mark">
                            WA
                        </div>

                        <div class="empty-title">
                            Welcome to Wentworth Academy
                        </div>

                        <div class="empty-description">
                            The Student Support Assistant can help with
                            academics, teachers, campus facilities, exams,
                            clubs, school rules, and general school
                            information.
                        </div>

                    </div>

                </div>

            </div>


            <div class="input-section">

                <form id="chat-form">

                    <div class="input-container">

                        <input
                            id="message-input"
                            type="text"
                            autocomplete="off"
                            placeholder="Ask about Wentworth Academy..."
                        >

                        <button
                            type="submit"
                            class="send-button"
                            id="send-button">
                            Send
                        </button>

                    </div>

                </form>

                <div class="disclaimer">
                    Wentworth Academy is a fictional school. Responses are generated by the project's NLP system.
                </div>

            </div>

        </div>

    </main>

</div>


<script>

    /* =========================================================
       DOM elements
       ========================================================= */

    const chatArea = document.getElementById("chat-area");
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");
    const clearButton = document.getElementById("clear-button");
    const mobileMenu = document.getElementById("mobile-menu");
    const sidebar = document.getElementById("sidebar");


    /* =========================================================
       Utility functions
       ========================================================= */

    function scrollToBottom() {
        chatArea.scrollTop = chatArea.scrollHeight;
    }


    function getTimeString() {

        return new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        });

    }


    function escapeHtml(text) {

        const div = document.createElement("div");

        div.textContent = text;

        return div.innerHTML;
    }


    function removeEmptyState() {

        const emptyState =
            document.getElementById("empty-state");

        if (emptyState) {
            emptyState.remove();
        }

    }


    function createEmptyState() {

        return `
            <div class="empty-state" id="empty-state">

                <div class="empty-content">

                    <div class="empty-mark">
                        WA
                    </div>

                    <div class="empty-title">
                        Welcome to Wentworth Academy
                    </div>

                    <div class="empty-description">
                        The Student Support Assistant can help with
                        academics, teachers, campus facilities, exams,
                        clubs, school rules, and general school
                        information.
                    </div>

                </div>

            </div>
        `;

    }


    /* =========================================================
       Add message
       ========================================================= */

    function addMessage(
        sender,
        message,
        intent = null,
        confidence = null
    ) {

        removeEmptyState();

        const messageElement =
            document.createElement("div");

        messageElement.className =
            "message " + sender;

        let meta = "";

        if (sender === "assistant") {

            let confidenceText = "";

            if (
                confidence !== null &&
                confidence !== undefined
            ) {

                confidenceText =
                    `<span class="confidence">
                        Confidence ${(confidence * 100).toFixed(0)}%
                    </span>`;

            }

            meta = `
                <div class="message-meta">

                    <span>${getTimeString()}</span>

                    ${confidenceText}

                </div>
            `;

        } else {

            meta = `
                <div class="message-meta">
                    <span>${getTimeString()}</span>
                </div>
            `;

        }


        const label =
            sender === "user"
                ? "You"
                : "Wentworth Assistant";


        messageElement.innerHTML = `

            <div class="message-inner">

                <div class="message-label">
                    ${label}
                </div>

                <div class="message-bubble">
                    ${escapeHtml(message)}
                </div>

                ${meta}

            </div>

        `;


        chatArea.appendChild(messageElement);

        scrollToBottom();

    }


    /* =========================================================
       Typing indicator
       ========================================================= */

    function showTyping() {

        removeTyping();

        const typingElement =
            document.createElement("div");

        typingElement.className =
            "message assistant";

        typingElement.id =
            "typing-indicator";

        typingElement.innerHTML = `

            <div class="message-inner">

                <div class="message-label">
                    Wentworth Assistant
                </div>

                <div class="message-bubble">

                    <div class="typing">

                        <span></span>
                        <span></span>
                        <span></span>

                    </div>

                </div>

            </div>

        `;

        chatArea.appendChild(typingElement);

        scrollToBottom();

    }


    function removeTyping() {

        const typing =
            document.getElementById(
                "typing-indicator"
            );

        if (typing) {
            typing.remove();
        }

    }


    /* =========================================================
       Send message
       ========================================================= */

    async function sendMessage(message) {

        message = message.trim();

        if (!message) {
            return;
        }

        addMessage("user", message);

        messageInput.value = "";

        sendButton.disabled = true;

        showTyping();


        try {

            const response =
                await fetch("/api/chat", {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        message: message
                    })

                });


            const data =
                await response.json();


            removeTyping();


            if (!response.ok) {

                addMessage(
                    "assistant",
                    data.error ||
                    "Sorry, something went wrong."
                );

                return;
            }


            addMessage(
                "assistant",
                data.response,
                data.intent,
                data.confidence
            );


        } catch (error) {

            console.error(error);

            removeTyping();

            addMessage(
                "assistant",
                "Sorry, I could not connect to the chatbot server."
            );

        } finally {

            sendButton.disabled = false;

            messageInput.focus();

        }

    }


    /* =========================================================
       Example buttons
       ========================================================= */

    document
        .querySelectorAll(".example-button")
        .forEach(button => {

            button.addEventListener(
                "click",
                () => {

                    const prompt =
                        button.dataset.prompt;

                    if (prompt) {

                        sendMessage(prompt);

                    }

                    if (
                        window.innerWidth <= 850
                    ) {

                        sidebar.classList.remove(
                            "open"
                        );

                    }

                }
            );

        });


    /* =========================================================
       Form submit
       ========================================================= */

    chatForm.addEventListener(
        "submit",
        event => {

            event.preventDefault();

            sendMessage(
                messageInput.value
            );

        }
    );


    /* =========================================================
       Clear conversation
       ========================================================= */

    clearButton.addEventListener(
        "click",
        () => {

            chatArea.innerHTML =
                createEmptyState();

            messageInput.value = "";

            messageInput.focus();

        }
    );


    /* =========================================================
       Mobile menu
       ========================================================= */

    mobileMenu.addEventListener(
        "click",
        () => {

            sidebar.classList.toggle(
                "open"
            );

        }
    );


    /* =========================================================
       Close sidebar when clicking main area on mobile
       ========================================================= */

    document
        .querySelector(".main")
        .addEventListener(
            "click",
            () => {

                if (
                    window.innerWidth <= 850
                ) {

                    sidebar.classList.remove(
                        "open"
                    );

                }

            }
        );


    /* =========================================================
       Initial focus
       ========================================================= */

    window.addEventListener(
        "DOMContentLoaded",
        () => {

            messageInput.focus();

        }
    );

</script>

</body>
</html>
"""


# ============================================================
# HTTP Request Handler
# ============================================================

class ChatbotHandler(BaseHTTPRequestHandler):

    def log_message(self, format_string, *args):
        print(
            f"[SERVER] {format_string % args}"
        )

    def send_json(self, data, status_code=200):

        response = json.dumps(
            data
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(response))
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.end_headers()

        self.wfile.write(response)


    def send_html(self):

        response = HTML_CONTENT.encode(
            "utf-8"
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(response))
        )

        self.end_headers()

        self.wfile.write(response)


    def do_GET(self):

        parsed_path = urlparse(
            self.path
        ).path


        if parsed_path in [
            "/",
            "/index.html"
        ]:

            self.send_html()

            return


        if parsed_path == "/api/status":

            self.send_json({

                "status": "online",

                "model": "MLP Neural Network",

                "examples":
                    DATA_STATS["examples"],

                "intents":
                    DATA_STATS["intents"]

            })

            return


        self.send_json(
            {
                "error": "Not found"
            },
            404
        )


    def do_POST(self):

        parsed_path = urlparse(self.path).path

        if parsed_path != "/api/chat":
            self.send_json(
                {"error": "Not found"},
                404
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            post_body = self.rfile.read(
                content_length
            )

            data = json.loads(
                post_body.decode("utf-8")
            )

            message = data.get(
                "message",
                ""
            )

            if not isinstance(message, str):
                self.send_json(
                    {"error": "Message must be text."},
                    400
                )
                return

            message = message.strip()

            if not message:
                self.send_json(
                    {"error": "Please enter a message."},
                    400
                )
                return

            # Predict intent
            intent, confidence = predict_intent(
                message,
                MODEL,
                PREPROCESSOR
            )

            # Generate response
            bot_response = get_response(
                intent,
                message
            )

            self.send_json({
                "response": bot_response,
                "intent": intent,
                "confidence": float(confidence)
            })

        except Exception as error:

            print("\n[ERROR]")
            print(str(error))

            self.send_json(
                {
                    "error":
                    "An error occurred while processing your message."
                },
                500
            )


# ============================================================
# Model initialization
# ============================================================

def initialize_model():

    global MODEL
    global PREPROCESSOR
    global DATA_STATS

    print("\n" + "=" * 60)
    print("Wentworth Academy NLP Chatbot")
    print("Initializing MLP model...")
    print("=" * 60)

    # Load the dataset
    df = load_intents(DATA_PATH)

    # Add features
    df = add_text_features(df)

    DATA_STATS["examples"] = len(df)
    DATA_STATS["intents"] = df["intent"].nunique()

    # Train the existing MLP model
    (
        MODEL,
        PREPROCESSOR,
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        X_test
    ) = train_mlp(df)

    print("\nMLP model initialized successfully.")

    print(
        f"Training examples: {len(y_train)}"
    )

    print(
        f"Test examples: {len(y_test)}"
    )

    print(
        f"Number of intents: {DATA_STATS['intents']}"
    )

    print("=" * 60)


# ============================================================
# Start server
# ============================================================

def start_server():

    server_address = (
        HOST,
        PORT
    )

    server = ThreadingHTTPServer(
        server_address,
        ChatbotHandler
    )


    url = (
        f"http://{HOST}:{PORT}"
    )


    print(
        f"\nServer running at: {url}"
    )

    print(
        "Press Ctrl+C to stop the application."
    )


    # Open browser shortly after
    # the server starts.
    threading.Timer(
        1.0,
        lambda: webbrowser.open(url)
    ).start()


    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "\n\nStopping server..."
        )

        server.shutdown()

        server.server_close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    try:

        initialize_model()

        start_server()

    except FileNotFoundError:

        print(
            "\nERROR: Could not find the dataset."
        )

        print(
            f"Expected file:"
        )

        print(
            DATA_PATH
        )

        sys.exit(1)


    except Exception as error:

        print(
            "\nERROR while starting application:"
        )

        print(
            str(error)
        )

        sys.exit(1)