<?php
/**
 * Serves a complete JPEG from feed.jpg (atomically updated by bridge.ps1).
 * Rejects truncated/corrupt frames and keeps the last good frame.
 */

declare(strict_types=1);

$config = require __DIR__ . '/config.php';
$feed = __DIR__ . '/feed.jpg';
$good = __DIR__ . '/feed.ok.jpg';
$minBytes = 20000;

function is_complete_jpeg(string $body, int $minBytes): bool
{
    if (strlen($body) < $minBytes) {
        return false;
    }
    // SOI marker
    if ($body[0] !== "\xFF" || $body[1] !== "\xD8") {
        return false;
    }
    // EOI marker near end (allow trailing nulls)
    $tail = rtrim($body, "\0");
    $len = strlen($tail);
    if ($len < 4) {
        return false;
    }
    return $tail[$len - 2] === "\xFF" && $tail[$len - 1] === "\xD9";
}

function send_jpeg(string $body): void
{
    header('Content-Type: image/jpeg');
    header('Cache-Control: no-store, no-cache, must-revalidate');
    header('Pragma: no-cache');
    header('X-Content-Type-Options: nosniff');
    header('Content-Length: ' . strlen($body));
    echo $body;
}

function send_error(int $status, string $error, string $hint): void
{
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    header('Cache-Control: no-store');
    echo json_encode([
        'ok' => false,
        'status' => $status,
        'error' => $error,
        'hint' => $hint,
    ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
}

$body = '';
if (is_file($feed)) {
    $body = (string) file_get_contents($feed);
}

if (is_complete_jpeg($body, $minBytes)) {
    // Refresh durable good frame (best-effort; ignore races).
    if (!is_file($good) || (filesize($good) !== strlen($body))) {
        @file_put_contents($good, $body, LOCK_EX);
    }
    send_jpeg($body);
    exit;
}

if (is_file($good)) {
    $fallback = (string) file_get_contents($good);
    if (is_complete_jpeg($fallback, $minBytes)) {
        header('X-Feed-Fallback: 1');
        send_jpeg($fallback);
        exit;
    }
}

send_error(
    503,
    'Kamera tasviri yo‘q yoki buzilgan',
    'Kompyuterda start-bridge.bat ni ishga tushiring va bir necha soniya kuting.'
);
