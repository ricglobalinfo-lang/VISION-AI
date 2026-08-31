<?php
/**
 * Camera settings for camera streaming
 *
 * try_live: set true only if the hosting server can reach the camera
 * (public IP / port-forward / SSH reverse tunnel to 127.0.0.1).
 * Default false — use bridge.ps1 to upload feed.jpg from the LAN.
 */
return [
    'try_live' => false,
    // If you set up: ssh -N -R 127.0.0.1:17220:192.168.1.100:80 user@example.uz
    // then use 'http://127.0.0.1:17220' and try_live => true
    'camera_base' => 'http://192.168.1.100',
    'snapshot_path' => '/ISAPI/Streaming/channels/101/picture',
    'username' => 'admin',
    'password' => 'YOUR_CAMERA_PASSWORD',
    'cache_seconds' => 0,
];
