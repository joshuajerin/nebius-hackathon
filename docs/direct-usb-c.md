# Direct USB-C diagnostic link

USB-C specifies a connector, not a diagnostic protocol. The robot computer and a normal industrial PC are usually both USB hosts and cannot be connected as though one were a peripheral.

The v1 workbench must expose a dedicated USB-C **device-mode** endpoint backed by a Linux USB gadget or equivalent controller. It should enumerate:

- CDC-NCM or CDC-ECM networking for the diagnostic API;
- CDC-ACM serial as a recovery console;
- no mass-storage or keyboard interfaces.

The physical link must use a fixed link-local subnet, disable forwarding to the plant network, authenticate the mission and asset identity, and close the diagnostic listener immediately when connector contact is lost. The included target emulator models this by keeping port `8766` physically closed until `/v1/dock` creates a short-lived link epoch. Mutations require an ordered sequence and idempotency key, and a crash with a pending mutation reconciles to `UNKNOWN` rather than being replayed.

The physical USB monitor should call the same control transition only after all of these are true:

1. port-depth/contact switch is active;
2. axial force remains below the connector-specific limit;
3. the USB device VID/PID and serial number match the asset registry;
4. the link-local interface is up;
5. the diagnostic endpoint returns the expected asset identity.

Removal or lease expiry must close the diagnostic listener and invalidate the epoch. Never expose a safety-reset or safety-bypass command through this channel.
