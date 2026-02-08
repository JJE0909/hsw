// widget_solver.js
const { webcrypto } = require('crypto');
const https = require('https');
const http = require('http');
const vm = require('vm');
const fs = require('fs');
const path = require('path');

let input = '';
process.stdin.on('data', d => input += d);
process.stdin.on('end', async () => {
    try {
        const config = JSON.parse(input.trim());
        const result = await solve(config);
        process.stdout.write(JSON.stringify(result));
    } catch (e) {
        process.stdout.write(JSON.stringify({
            error: e.message,
            stack: e.stack?.split('\n').slice(0, 8)
        }));
        process.exit(1);
    }
});

function makeRequest(urlStr, options = {}) {
    return new Promise((resolve, reject) => {
        const url = new URL(urlStr);
        const mod = url.protocol === 'https:' ? https : http;
        const reqOpts = {
            hostname: url.hostname,
            port: url.port || (url.protocol === 'https:' ? 443 : 80),
            path: url.pathname + url.search,
            method: options.method || 'GET',
            headers: options.headers || {},
        };
        const req = mod.request(reqOpts, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve({
                status: res.statusCode,
                headers: res.headers,
                body: data,
                url: urlStr,
            }));
        });
        req.on('error', reject);
        if (options.body) req.write(options.body);
        req.end();
    });
}

function createElement(tag) {
    const el = {
        tagName: (tag || 'DIV').toUpperCase(),
        nodeName: (tag || 'DIV').toUpperCase(),
        nodeType: 1,
        style: new Proxy({}, { get: () => '', set: () => true }),
        classList: {
            add() {}, remove() {}, contains() { return false; }, toggle() {},
        },
        children: [],
        childNodes: [],
        parentNode: null,
        parentElement: null,
        ownerDocument: null, // set later
        attributes: [],
        dataset: {},
        innerHTML: '',
        outerHTML: '',
        textContent: '',
        innerText: '',
        id: '',
        className: '',
        src: '',
        href: '',
        type: '',
        value: '',
        checked: false,
        disabled: false,
        width: 300,
        height: 150,
        offsetWidth: 300,
        offsetHeight: 150,
        clientWidth: 300,
        clientHeight: 150,
        scrollWidth: 300,
        scrollHeight: 150,
        offsetTop: 0,
        offsetLeft: 0,
        scrollTop: 0,
        scrollLeft: 0,
        setAttribute(k, v) { el[k] = v; },
        getAttribute(k) { return el[k] !== undefined ? String(el[k]) : null; },
        hasAttribute(k) { return el[k] !== undefined; },
        removeAttribute() {},
        appendChild(c) { if (c) { el.children.push(c); el.childNodes.push(c); c.parentNode = el; c.parentElement = el; } return c; },
        removeChild(c) { return c; },
        insertBefore(c) { if (c) { el.children.unshift(c); el.childNodes.unshift(c); } return c; },
        replaceChild(n, o) { return o; },
        cloneNode() { return createElement(tag); },
        contains() { return false; },
        querySelector() { return null; },
        querySelectorAll() { return []; },
        getElementsByTagName() { return []; },
        getElementsByClassName() { return []; },
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() { return true; },
        getBoundingClientRect() {
            return { top: 0, left: 0, bottom: 0, right: 0, width: 300, height: 150, x: 0, y: 0 };
        },
        getContext(type) {
            if (type === '2d') {
                return {
                    canvas: el,
                    fillStyle: '', strokeStyle: '', font: '10px sans-serif',
                    textAlign: 'start', textBaseline: 'alphabetic',
                    globalAlpha: 1, globalCompositeOperation: 'source-over',
                    imageSmoothingEnabled: true, lineCap: 'butt', lineJoin: 'miter',
                    lineWidth: 1, miterLimit: 10, shadowBlur: 0, shadowColor: 'rgba(0,0,0,0)',
                    shadowOffsetX: 0, shadowOffsetY: 0,
                    fillRect() {}, clearRect() {}, strokeRect() {},
                    fillText() {}, strokeText() {},
                    measureText(t) { return { width: (t || '').length * 7 }; },
                    getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
                    putImageData() {}, createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4), width: w, height: h }; },
                    drawImage() {},
                    beginPath() {}, closePath() {}, moveTo() {}, lineTo() {},
                    arc() {}, arcTo() {}, rect() {}, ellipse() {},
                    fill() {}, stroke() {}, clip() {},
                    save() {}, restore() {}, translate() {}, rotate() {}, scale() {},
                    transform() {}, setTransform() {}, resetTransform() {},
                    createLinearGradient() { return { addColorStop() {} }; },
                    createRadialGradient() { return { addColorStop() {} }; },
                    createPattern() { return {}; },
                    getLineSash() { return []; }, setLineDash() {},
                    isPointInPath() { return false; },
                    toDataURL() { return 'data:image/png;base64,iVBORw0KGgo='; },
                };
            }
            if (type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl') {
                const glConsts = {
                    VENDOR: 7936, RENDERER: 7937, VERSION: 7938, SHADING_LANGUAGE_VERSION: 35724,
                    MAX_TEXTURE_SIZE: 3379, MAX_RENDERBUFFER_SIZE: 34024,
                    UNMASKED_VENDOR_WEBGL: 37445, UNMASKED_RENDERER_WEBGL: 37446,
                };
                return {
                    getParameter(p) {
                        if (p === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce GTX 1070 Direct3D11 vs_5_0 ps_5_0)';
                        if (p === 37445) return 'Google Inc. (NVIDIA)';
                        if (p === 7937) return 'WebKit WebGL';
                        if (p === 7936) return 'WebKit';
                        if (p === 7938) return type === 'webgl2' ? 'WebGL 2.0' : 'WebGL 1.0';
                        if (p === 35724) return type === 'webgl2' ? 'WebGL GLSL ES 3.00' : 'WebGL GLSL ES 1.0';
                        if (p === 3379 || p === 34024) return 16384;
                        return 0;
                    },
                    getExtension(n) {
                        if (n === 'WEBGL_debug_renderer_info') return glConsts;
                        return null;
                    },
                    getSupportedExtensions() { return ['WEBGL_debug_renderer_info']; },
                    createShader() { return {}; }, shaderSource() {}, compileShader() {},
                    getShaderParameter() { return true; }, createProgram() { return {}; },
                    attachShader() {}, linkProgram() {}, getProgramParameter() { return true; },
                    useProgram() {}, createBuffer() { return {}; }, bindBuffer() {},
                    bufferData() {}, enableVertexAttribArray() {}, vertexAttribPointer() {},
                    getAttribLocation() { return 0; }, getUniformLocation() { return {}; },
                    uniform1f() {}, uniform2f() {}, uniform3f() {}, uniform4f() {},
                    drawArrays() {}, drawElements() {}, viewport() {}, clear() {},
                    clearColor() {}, enable() {}, disable() {}, blendFunc() {},
                    createTexture() { return {}; }, bindTexture() {}, texImage2D() {},
                    texParameteri() {}, activeTexture() {}, deleteTexture() {},
                    deleteBuffer() {}, deleteShader() {}, deleteProgram() {},
                    getShaderInfoLog() { return ''; }, getProgramInfoLog() { return ''; },
                    createFramebuffer() { return {}; }, bindFramebuffer() {},
                    framebufferTexture2D() {}, checkFramebufferStatus() { return 36053; },
                    readPixels(x, y, w, h, fmt, type, pixels) {},
                    isContextLost() { return false; },
                    canvas: el,
                    drawingBufferWidth: 300, drawingBufferHeight: 150,
                };
            }
            return null;
        },
        toDataURL() { return 'data:image/png;base64,iVBORw0KGgo='; },
        focus() {}, blur() {}, click() {}, remove() {},
        toString() { return `[object HTML${el.tagName.charAt(0) + el.tagName.slice(1).toLowerCase()}Element]`; },
    };
    return el;
}

function createDocument() {
    const html = createElement('html');
    const head = createElement('head');
    const body = createElement('body');
    html.appendChild(head);
    html.appendChild(body);

    const doc = {
        nodeType: 9,
        documentElement: html,
        head,
        body,
        readyState: 'complete',
        visibilityState: 'visible',
        hidden: false,
        cookie: '',
        domain: 'newassets.hcaptcha.com',
        referrer: '',
        title: '',
        URL: 'https://newassets.hcaptcha.com/captcha/v1/b1129b9/static/hcaptcha.html',
        documentURI: 'https://newassets.hcaptcha.com/captcha/v1/b1129b9/static/hcaptcha.html',
        location: null, // set later
        defaultView: null, // set later
        compatMode: 'CSS1Compat',
        characterSet: 'UTF-8',
        charset: 'UTF-8',
        contentType: 'text/html',
        createElement(tag) {
            const el = createElement(tag);
            el.ownerDocument = doc;
            return el;
        },
        createElementNS(ns, tag) { return doc.createElement(tag); },
        createTextNode(t) { return { nodeType: 3, textContent: t, data: t }; },
        createComment(t) { return { nodeType: 8, textContent: t, data: t }; },
        createDocumentFragment() {
            return {
                nodeType: 11, children: [], childNodes: [],
                appendChild(c) { this.children.push(c); this.childNodes.push(c); return c; },
                removeChild(c) { return c; },
                querySelector() { return null; },
                querySelectorAll() { return []; },
            };
        },
        createEvent(type) {
            return { type: '', initEvent(t) { this.type = t; }, preventDefault() {}, stopPropagation() {} };
        },
        createRange() {
            return {
                setStart() {}, setEnd() {}, collapse() {},
                selectNode() {}, selectNodeContents() {},
                getBoundingClientRect() { return { top: 0, left: 0, bottom: 0, right: 0, width: 0, height: 0 }; },
                createContextualFragment(html) { return doc.createDocumentFragment(); },
            };
        },
        createTreeWalker() {
            return { nextNode() { return null; }, currentNode: null };
        },
        getElementById() { return null; },
        querySelector(sel) {
            if (sel === 'head') return head;
            if (sel === 'body') return body;
            if (sel === 'html') return html;
            return null;
        },
        querySelectorAll(sel) {
            const r = doc.querySelector(sel);
            return r ? [r] : [];
        },
        getElementsByTagName(t) {
            t = t.toLowerCase();
            if (t === 'head') return [head];
            if (t === 'body') return [body];
            if (t === 'html') return [html];
            if (t === 'script') return [];
            if (t === '*') return [html, head, body];
            return [];
        },
        getElementsByClassName() { return []; },
        getElementsByName() { return []; },
        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() { return true; },
        hasFocus() { return true; },
        execCommand() { return false; },
        getSelection() { return { rangeCount: 0, addRange() {}, removeAllRanges() {} }; },
        adoptNode(n) { return n; },
        importNode(n) { return n; },
        write() {},
        writeln() {},
        open() {},
        close() {},
    };

    // Cross-reference
    html.ownerDocument = doc;
    head.ownerDocument = doc;
    body.ownerDocument = doc;

    return doc;
}

function solveHSW(hswCode, req) {
    return new Promise(async (resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('HSW timed out after 30s')), 30000);
        const messageListeners = [];
        let resolved = false;

        function doResolve(data) {
            if (resolved) return;
            resolved = true;
            clearTimeout(timeout);
            if (typeof data === 'string') {
                resolve(data);
            } else if (data && typeof data === 'object') {
                resolve(data.result || data.token || data.proof || data.answer || JSON.stringify(data));
            } else {
                resolve(String(data));
            }
        }

        const doc = createDocument();

        const locationObj = {
            href: 'https://newassets.hcaptcha.com/captcha/v1/b1129b9/static/hcaptcha.html',
            origin: 'https://newassets.hcaptcha.com',
            protocol: 'https:',
            host: 'newassets.hcaptcha.com',
            hostname: 'newassets.hcaptcha.com',
            pathname: '/captcha/v1/b1129b9/static/hcaptcha.html',
            port: '', search: '', hash: '',
            assign() {}, reload() {}, replace() {},
            toString() { return this.href; },
        };

        doc.location = locationObj;

        const contextObj = {
            document: doc,
            location: locationObj,
            navigator: {
                userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                language: 'en-US', languages: ['en-US', 'en'], platform: 'Win32',
                hardwareConcurrency: 8, deviceMemory: 8, maxTouchPoints: 0,
                onLine: true, vendor: 'Google Inc.', webdriver: false,
                cookieEnabled: true,
                plugins: { length: 0, item() { return null; }, namedItem() { return null; }, refresh() {} },
                mimeTypes: { length: 0, item() { return null; }, namedItem() { return null; } },
                doNotTrack: null,
                getBattery: async () => ({ charging: true, chargingTime: 0, dischargingTime: Infinity, level: 1, addEventListener() {} }),
                connection: { effectiveType: '4g', rtt: 50, downlink: 10, saveData: false, addEventListener() {} },
                mediaDevices: { enumerateDevices: async () => [], addEventListener() {} },
                permissions: { query: async () => ({ state: 'prompt', addEventListener() {} }) },
                clipboard: { readText: async () => '', writeText: async () => {} },
                locks: { request: async () => {} },
                storage: { estimate: async () => ({ quota: 0, usage: 0 }), persist: async () => false },
                serviceWorker: { ready: new Promise(() => {}), register: async () => ({}), addEventListener() {} },
                sendBeacon() { return true; },
                userAgentData: {
                    brands: [{ brand: 'Chromium', version: '131' }, { brand: 'Not_A Brand', version: '24' }],
                    mobile: false, platform: 'Windows',
                    getHighEntropyValues: async () => ({
                        architecture: 'x86', bitness: '64', model: '', platformVersion: '15.0.0',
                        fullVersionList: [{ brand: 'Chromium', version: '131.0.6778.86' }],
                    }),
                },
            },
            screen: {
                width: 1920, height: 1080, availWidth: 1920, availHeight: 1040,
                colorDepth: 24, pixelDepth: 24, availLeft: 0, availTop: 0,
                orientation: { type: 'landscape-primary', angle: 0, addEventListener() {} },
            },
            history: { length: 1, state: null, pushState() {}, replaceState() {}, go() {}, back() {}, forward() {} },
            crypto: webcrypto,
            console: { log() {}, warn() {}, error() {}, info() {}, debug() {}, trace() {}, dir() {}, time() {}, timeEnd() {}, table() {}, group() {}, groupEnd() {} },
            performance: {
                now: (() => { const s = Date.now(); return () => Date.now() - s; })(),
                timeOrigin: Date.now() - 5000,
                timing: { navigationStart: Date.now() - 5000, fetchStart: Date.now() - 4900, domContentLoadedEventEnd: Date.now() - 3000, loadEventEnd: Date.now() - 2500 },
                memory: { jsHeapSizeLimit: 4294705152, totalJSHeapSize: 35000000, usedJSHeapSize: 25000000 },
                getEntries() { return []; }, getEntriesByType() { return []; }, getEntriesByName() { return []; },
                mark() {}, measure() {}, clearMarks() {}, clearMeasures() {},
            },
            localStorage: { _d: {}, getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; }, clear() { this._d = {}; }, get length() { return Object.keys(this._d).length; } },
            sessionStorage: { _d: {}, getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; }, clear() { this._d = {}; }, get length() { return Object.keys(this._d).length; } },
            setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask,
            requestAnimationFrame: (cb) => setTimeout(cb, 16),
            cancelAnimationFrame: clearTimeout,
            requestIdleCallback: (cb) => setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 50 }), 1),
            cancelIdleCallback: clearTimeout,
            atob: (s) => Buffer.from(s, 'base64').toString('binary'),
            btoa: (s) => Buffer.from(s, 'binary').toString('base64'),
            TextEncoder, TextDecoder, URL, URLSearchParams,
            Uint8Array, Int32Array, Float64Array, Float32Array, Uint32Array, Uint16Array,
            Int8Array, Int16Array, BigInt64Array, BigUint64Array, Uint8ClampedArray,
            ArrayBuffer, DataView, SharedArrayBuffer,
            Map, Set, WeakMap, WeakSet, WeakRef, FinalizationRegistry,
            Promise, Proxy, Reflect, Symbol, BigInt, WebAssembly,
            Math, Date, JSON, String, Number, Boolean, Array, Object, RegExp, Function,
            Error, TypeError, RangeError, ReferenceError, SyntaxError, URIError, EvalError,
            parseInt, parseFloat, isNaN, isFinite, NaN, Infinity, undefined,
            encodeURIComponent, decodeURIComponent, encodeURI, decodeURI,
            escape, unescape,
            innerWidth: 1920, innerHeight: 969, outerWidth: 1920, outerHeight: 1040,
            devicePixelRatio: 1, screenX: 0, screenY: 0, screenLeft: 0, screenTop: 0,
            pageXOffset: 0, pageYOffset: 0, scrollX: 0, scrollY: 0,
            visualViewport: { width: 1920, height: 969, offsetTop: 0, offsetLeft: 0, scale: 1, addEventListener() {} },
            origin: 'https://newassets.hcaptcha.com',
            isSecureContext: true,
            crossOriginIsolated: false,
            postMessage: function(data) { doResolve(data); },
            addEventListener: function(type, fn) { if (type === 'message') messageListeners.push(fn); },
            removeEventListener: function() {},
            close: function() {},
            importScripts: function() {},
            fetch: async (url, opts) => {
                const urlStr = typeof url === 'string' ? url : url?.url || String(url);
                const resp = await makeRequest(urlStr, {
                    method: opts?.method || 'GET',
                    headers: opts?.headers || {},
                    body: opts?.body,
                });
                const bodyText = resp.body;
                return {
                    ok: resp.status >= 200 && resp.status < 300,
                    status: resp.status, statusText: resp.status === 200 ? 'OK' : String(resp.status),
                    url: urlStr,
                    headers: {
                        get: (k) => resp.headers[k.toLowerCase()] || null,
                        has: (k) => k.toLowerCase() in resp.headers,
                        forEach: (cb) => Object.entries(resp.headers).forEach(([k, v]) => cb(v, k)),
                        entries: () => Object.entries(resp.headers),
                    },
                    json: async () => JSON.parse(bodyText),
                    text: async () => bodyText,
                    arrayBuffer: async () => {
                        const buf = Buffer.from(bodyText, 'binary');
                        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                    },
                    blob: async () => ({ size: bodyText.length, type: resp.headers['content-type'] || '' }),
                    clone() { return this; },
                    body: null, bodyUsed: false, type: 'cors', redirected: false,
                };
            },
            XMLHttpRequest: createXHRClass(),
            Blob: class Blob {
                constructor(parts = [], opts = {}) {
                    this._parts = parts; this.type = opts.type || '';
                    this.size = parts.reduce((s, p) => s + (p?.length || p?.byteLength || 0), 0);
                }
                async text() { return this._parts.map(p => typeof p === 'string' ? p : '').join(''); }
                async arrayBuffer() {
                    const bufs = this._parts.map(p => {
                        if (p instanceof ArrayBuffer) return Buffer.from(p);
                        if (ArrayBuffer.isView(p)) return Buffer.from(p.buffer, p.byteOffset, p.byteLength);
                        return Buffer.from(String(p));
                    });
                    const c = Buffer.concat(bufs);
                    return c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength);
                }
                slice(start, end, type) { return new Blob([], { type }); }
            },
            File: class File { constructor(parts, name, opts) { this.name = name; this.type = opts?.type || ''; this.size = 0; } },
            FileReader: class FileReader {
                readAsDataURL() { setTimeout(() => this.onload?.({ target: { result: 'data:;base64,' } }), 0); }
                readAsArrayBuffer() { setTimeout(() => this.onload?.({ target: { result: new ArrayBuffer(0) } }), 0); }
                readAsText() { setTimeout(() => this.onload?.({ target: { result: '' } }), 0); }
                addEventListener(t, fn) { if (t === 'load') this.onload = fn; }
            },
            FormData: class FormData {
                constructor() { this._d = []; }
                append(k, v) { this._d.push([k, v]); }
                get(k) { const e = this._d.find(([kk]) => kk === k); return e ? e[1] : null; }
                getAll(k) { return this._d.filter(([kk]) => kk === k).map(([, v]) => v); }
                has(k) { return this._d.some(([kk]) => kk === k); }
                delete(k) { this._d = this._d.filter(([kk]) => kk !== k); }
                forEach(cb) { this._d.forEach(([k, v]) => cb(v, k)); }
                entries() { return this._d[Symbol.iterator](); }
                keys() { return this._d.map(([k]) => k)[Symbol.iterator](); }
                values() { return this._d.map(([, v]) => v)[Symbol.iterator](); }
            },
            Worker: class Worker { constructor() { this.onmessage = null; } postMessage() {} terminate() {} addEventListener() {} removeEventListener() {} },
            MessageChannel: class MessageChannel {
                constructor() {
                    this.port1 = { postMessage() {}, addEventListener() {}, start() {}, close() {} };
                    this.port2 = { postMessage() {}, addEventListener() {}, start() {}, close() {} };
                }
            },
            BroadcastChannel: class BroadcastChannel { constructor() { this.onmessage = null; } postMessage() {} close() {} addEventListener() {} },
            Event: class Event {
                constructor(t, i) { this.type = t; this.bubbles = i?.bubbles || false; this.cancelable = i?.cancelable || false; this.defaultPrevented = false; }
                preventDefault() { this.defaultPrevented = true; } stopPropagation() {} stopImmediatePropagation() {}
            },
            CustomEvent: class CustomEvent { constructor(t, i) { this.type = t; this.detail = i?.detail; } preventDefault() {} stopPropagation() {} },
            MessageEvent: class MessageEvent { constructor(t, i) { this.type = t; this.data = i?.data; this.origin = i?.origin || ''; } },
            ErrorEvent: class ErrorEvent { constructor(t, i) { this.type = t; this.message = i?.message; this.error = i?.error; } },
            MutationObserver: class MutationObserver { observe() {} disconnect() {} takeRecords() { return []; } },
            IntersectionObserver: class IntersectionObserver { observe() {} unobserve() {} disconnect() {} },
            ResizeObserver: class ResizeObserver { observe() {} unobserve() {} disconnect() {} },
            PerformanceObserver: class PerformanceObserver { observe() {} disconnect() {} static supportedEntryTypes = [] },
            Image: class Image {
                constructor(w, h) { this.width = w || 0; this.height = h || 0; this.src = ''; this.complete = true; this.naturalWidth = 300; this.naturalHeight = 150; this.onload = null; this.onerror = null; }
                addEventListener(t, fn) { if (t === 'load') this.onload = fn; if (t === 'error') this.onerror = fn; } removeEventListener() {}
            },
            DOMParser: class DOMParser { parseFromString() { return createDocument(); } },
            AbortController: class AbortController {
                constructor() { this.signal = { aborted: false, reason: undefined, addEventListener() {}, removeEventListener() {}, throwIfAborted() {} }; }
                abort(reason) { this.signal.aborted = true; this.signal.reason = reason; }
            },
            Headers: class Headers {
                constructor(init) { this._h = {}; if (init) { if (typeof init.forEach === 'function') init.forEach((v, k) => { this._h[k.toLowerCase()] = v; }); else Object.entries(init).forEach(([k, v]) => { this._h[k.toLowerCase()] = v; }); } }
                get(k) { return this._h[k.toLowerCase()] || null; } set(k, v) { this._h[k.toLowerCase()] = v; }
                has(k) { return k.toLowerCase() in this._h; } delete(k) { delete this._h[k.toLowerCase()]; }
                forEach(cb) { Object.entries(this._h).forEach(([k, v]) => cb(v, k)); }
            },
            Request: class Request { constructor(u, o) { this.url = u; this.method = o?.method || 'GET'; } },
            Response: class Response { constructor(b, i) { this.body = b; this.status = i?.status || 200; this.ok = this.status < 400; } },
            Element: class Element {}, HTMLElement: class HTMLElement {}, HTMLCanvasElement: class HTMLCanvasElement {},
            HTMLImageElement: class HTMLImageElement {}, HTMLVideoElement: class HTMLVideoElement {},
            HTMLScriptElement: class HTMLScriptElement {}, HTMLIFrameElement: class HTMLIFrameElement {},
            Node: class Node {}, NodeList: class NodeList {},
            DOMRect: class DOMRect { constructor(x, y, w, h) { this.x = x || 0; this.y = y || 0; this.width = w || 0; this.height = h || 0; } },
            DOMRectReadOnly: class DOMRectReadOnly { constructor(x, y, w, h) { this.x = x || 0; this.y = y || 0; this.width = w || 0; this.height = h || 0; } },
            CSS: { supports() { return false; }, escape(s) { return s; } },
            CSSStyleDeclaration: class CSSStyleDeclaration {},
            matchMedia: () => ({ matches: false, media: '', addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }),
            getComputedStyle: () => new Proxy({}, { get: (t, p) => typeof p === 'string' ? '' : undefined }),
            getSelection: () => ({ rangeCount: 0, addRange() {}, removeAllRanges() {}, toString() { return ''; } }),
            scrollTo() {}, scrollBy() {}, scroll() {}, focus() {}, blur() {},
            open() { return null; }, print() {}, stop() {},
            alert() {}, confirm() { return false; }, prompt() { return null; },
            indexedDB: {
                open() {
                    const req = { result: null, error: null, onsuccess: null, onerror: null, onupgradeneeded: null, addEventListener() {} };
                    setTimeout(() => req.onerror?.({ target: req }), 0);
                    return req;
                },
                databases: async () => [],
            },
            Notification: class Notification { static permission = 'denied'; static requestPermission() { return Promise.resolve('denied'); } },
            AudioContext: class AudioContext { constructor() { this.destination = {}; this.sampleRate = 44100; this.state = 'suspended'; } close() { return Promise.resolve(); } createOscillator() { return { connect() {}, start() {}, stop() {}, frequency: { value: 0 } }; } createDynamicsCompressor() { return { connect() {}, threshold: { value: 0 }, knee: { value: 0 }, ratio: { value: 0 }, reduction: { value: 0 }, attack: { value: 0 }, release: { value: 0 } }; } createGain() { return { connect() {}, gain: { value: 1 } }; } createAnalyser() { return { connect() {}, fftSize: 0, getFloatFrequencyData() {} }; } resume() { return Promise.resolve(); } },
            webkitAudioContext: undefined,
            OfflineAudioContext: class OfflineAudioContext { constructor() { this.destination = {}; } startRendering() { return Promise.resolve({ getChannelData() { return new Float32Array(0); } }); } createOscillator() { return { connect() {}, start() {}, stop() {}, frequency: { value: 0 } }; } createDynamicsCompressor() { return { connect() {}, threshold: { value: 0 }, knee: { value: 0 }, ratio: { value: 0 }, attack: { value: 0 }, release: { value: 0 } }; } },
            SpeechSynthesisUtterance: class SpeechSynthesisUtterance {},
            speechSynthesis: { getVoices() { return []; }, addEventListener() {} },
            module: { exports: {} },
            exports: {},
        };

        // Pre-define onmessage as writable+configurable
        Object.defineProperty(contextObj, 'onmessage', {
            value: null, writable: true, configurable: true, enumerable: true,
        });

        const ctx = vm.createContext(contextObj, {
            codeGeneration: { strings: true, wasm: true },
        });

        vm.runInContext(`
            self = this;
            window = this;
            globalThis = this;
            top = this;
            parent = this;
            frames = this;
            document.defaultView = this;
        `, ctx);

        vm.runInContext(`
            (function() {
                var _ml = [];
                self.__ml = _ml;
                var _origAEL = self.addEventListener;
                self.addEventListener = function(type, fn, opts) {
                    if (type === 'message') _ml.push(fn);
                    if (_origAEL) _origAEL.call(self, type, fn, opts);
                };
            })();
        `, ctx);

        try {
            const script = new vm.Script(hswCode, { filename: 'hsw.js', timeout: 30000 });
            script.runInContext(ctx);
        } catch (e) {
            clearTimeout(timeout);
            reject(new Error(`HSW load error: ${e.message}`));
            return;
        }

        // Diagnose what hsw.js created
        const diagScript = `
            (function() {
                var info = { newProps: [], fnProps: [], onmessageType: typeof onmessage };
                var dominated = ['document','location','navigator','screen','history','crypto',
                    'console','performance','localStorage','sessionStorage','setTimeout',
                    'clearTimeout','setInterval','clearInterval','queueMicrotask',
                    'requestAnimationFrame','cancelAnimationFrame','requestIdleCallback',
                    'cancelIdleCallback','atob','btoa','TextEncoder','TextDecoder','URL',
                    'URLSearchParams','fetch','XMLHttpRequest','Worker','MessageChannel',
                    'BroadcastChannel','Event','CustomEvent','MessageEvent','ErrorEvent',
                    'MutationObserver','IntersectionObserver','ResizeObserver',
                    'PerformanceObserver','Image','DOMParser','AbortController','Headers',
                    'Request','Response','Blob','File','FileReader','FormData',
                    'Element','HTMLElement','HTMLCanvasElement','HTMLImageElement',
                    'HTMLVideoElement','HTMLScriptElement','HTMLIFrameElement',
                    'Node','NodeList','DOMRect','DOMRectReadOnly','CSS',
                    'CSSStyleDeclaration','AudioContext','OfflineAudioContext',
                    'SpeechSynthesisUtterance','Notification',
                    'matchMedia','getComputedStyle','getSelection',
                    'scrollTo','scrollBy','scroll','focus','blur','open','print','stop',
                    'alert','confirm','prompt','indexedDB','postMessage','addEventListener',
                    'removeEventListener','close','importScripts','module','exports',
                    'self','window','globalThis','top','parent','frames','origin',
                    'isSecureContext','crossOriginIsolated','innerWidth','innerHeight',
                    'outerWidth','outerHeight','devicePixelRatio','screenX','screenY',
                    'screenLeft','screenTop','pageXOffset','pageYOffset','scrollX','scrollY',
                    'visualViewport','webkitAudioContext','speechSynthesis',
                    'Uint8Array','Int32Array','Float64Array','Float32Array','Uint32Array',
                    'Uint16Array','Int8Array','Int16Array','BigInt64Array','BigUint64Array',
                    'Uint8ClampedArray','ArrayBuffer','DataView','SharedArrayBuffer',
                    'Map','Set','WeakMap','WeakSet','WeakRef','FinalizationRegistry',
                    'Promise','Proxy','Reflect','Symbol','BigInt','WebAssembly',
                    'Math','Date','JSON','String','Number','Boolean','Array','Object',
                    'RegExp','Function','Error','TypeError','RangeError','ReferenceError',
                    'SyntaxError','URIError','EvalError','parseInt','parseFloat','isNaN',
                    'isFinite','NaN','Infinity','undefined','encodeURIComponent',
                    'decodeURIComponent','encodeURI','decodeURI','escape','unescape',
                    'onmessage','__ml'];
                var dominated_set = {};
                for (var i = 0; i < dominated.length; i++) dominated_set[dominated[i]] = 1;
                var keys = Object.getOwnPropertyNames(self);
                for (var j = 0; j < keys.length; j++) {
                    var k = keys[j];
                    if (dominated_set[k]) continue;
                    try {
                        var v = self[k];
                        var t = typeof v;
                        info.newProps.push({ name: k, type: t, isFunc: t === 'function', len: t === 'function' ? v.length : undefined });
                        if (t === 'function') info.fnProps.push(k);
                    } catch(e) {}
                }
                info.moduleExportsType = typeof module.exports;
                if (typeof module.exports === 'object' && module.exports !== null) {
                    info.moduleExportsKeys = Object.keys(module.exports);
                }
                if (typeof module.exports === 'function') {
                    info.moduleExportsIsFunc = true;
                    info.moduleExportsFnLen = module.exports.length;
                }
                info.exportsType = typeof exports;
                if (typeof exports === 'object' && exports !== null) {
                    info.exportsKeys = Object.keys(exports);
                }
                info.onmessageValue = String(onmessage).substring(0, 200);
                info.selfKeys = keys.length;
                return JSON.stringify(info);
            })()
        `;

        let diagResult;
        try {
            diagResult = JSON.parse(vm.runInContext(diagScript, ctx));
        } catch (e) {
            diagResult = { error: e.message };
        }
        process.stderr.write(`  [hsw] Diagnostic: ${JSON.stringify(diagResult).substring(0, 800)}\n`);

        // Collect message listeners
        const ctxOnmessage = vm.runInContext('typeof onmessage === "function" ? onmessage : null', ctx);
        if (ctxOnmessage) messageListeners.push(ctxOnmessage);

        const selfOnmessage = vm.runInContext('typeof self.onmessage === "function" ? self.onmessage : null', ctx);
        if (selfOnmessage && selfOnmessage !== ctxOnmessage) messageListeners.push(selfOnmessage);

        try {
            const internalListeners = vm.runInContext('self.__ml || []', ctx);
            for (const fn of internalListeners) {
                if (typeof fn === 'function' && !messageListeners.includes(fn)) messageListeners.push(fn);
            }
        } catch {}

        process.stderr.write(`  [hsw] Found ${messageListeners.length} message listener(s)\n`);

        // If hsw.js added new functions, try calling them
        if (diagResult.fnProps && diagResult.fnProps.length > 0) {
            process.stderr.write(`  [hsw] Trying new functions: ${diagResult.fnProps.join(', ')}\n`);
            for (const fnName of diagResult.fnProps) {
                if (resolved) break;
                try {
                    const result = await vm.runInContext(`
                        (async function() {
                            try {
                                var fn = self[${JSON.stringify(fnName)}];
                                var r = await fn(${JSON.stringify(req)});
                                return r;
                            } catch(e) {
                                return { __error: e.message };
                            }
                        })()
                    `, ctx);
                    process.stderr.write(`  [hsw] ${fnName}() returned: ${String(result).substring(0, 100)}\n`);
                    if (typeof result === 'string' && result.length > 10) {
                        doResolve(result);
                        return;
                    }
                    if (result && result.__error) {
                        process.stderr.write(`  [hsw] ${fnName}() inner error: ${result.__error}\n`);
                    }
                } catch (e) {
                    process.stderr.write(`  [hsw] ${fnName}() error: ${e.message}\n`);
                }
            }
        }

        // Wait for async initialization
        await new Promise(r => setTimeout(r, 1500));

        const laterOnmessage = vm.runInContext('typeof onmessage === "function" ? onmessage : null', ctx);
        if (laterOnmessage && !messageListeners.includes(laterOnmessage)) {
            messageListeners.push(laterOnmessage);
            process.stderr.write(`  [hsw] Found delayed onmessage handler!\n`);
        }

        const laterSelfOnmessage = vm.runInContext('typeof self.onmessage === "function" ? self.onmessage : null', ctx);
        if (laterSelfOnmessage && !messageListeners.includes(laterSelfOnmessage)) {
            messageListeners.push(laterSelfOnmessage);
            process.stderr.write(`  [hsw] Found delayed self.onmessage handler!\n`);
        }

        try {
            const laterInternal = vm.runInContext('self.__ml || []', ctx);
            for (const fn of laterInternal) {
                if (typeof fn === 'function' && !messageListeners.includes(fn)) {
                    messageListeners.push(fn);
                    process.stderr.write(`  [hsw] Found delayed __ml handler!\n`);
                }
            }
        } catch {}

        process.stderr.write(`  [hsw] After delay: ${messageListeners.length} message listener(s)\n`);

        if (messageListeners.length === 0) {
            // Try module.exports
            if (diagResult.moduleExportsIsFunc) {
                try {
                    const meFn = vm.runInContext('module.exports', ctx);
                    const r = await meFn(req);
                    if (r && typeof r === 'string' && r.length > 10) { doResolve(r); return; }
                } catch (e) {
                    process.stderr.write(`  [hsw] module.exports() error: ${e.message}\n`);
                }
            }
            if (diagResult.moduleExportsKeys && diagResult.moduleExportsKeys.length > 0) {
                const me = vm.runInContext('module.exports', ctx);
                for (const key of diagResult.moduleExportsKeys) {
                    if (typeof me[key] === 'function') {
                        try {
                            const r = await me[key](req);
                            if (r && typeof r === 'string' && r.length > 10) { doResolve(r); return; }
                        } catch (e) {
                            process.stderr.write(`  [hsw] module.exports.${key}() error: ${e.message}\n`);
                        }
                    }
                }
            }
            clearTimeout(timeout);
            reject(new Error('No message listeners and no callable functions found'));
            return;
        }

        // Send challenge via message listeners
        const formats = [
            { data: req },
            { data: { type: 'hsw', req: req } },
        ];

        for (const fmt of formats) {
            if (resolved) break;
            for (const fn of messageListeners) {
                if (resolved) break;
                try {
                    fn(fmt);
                } catch (e) {
                    process.stderr.write(`  [hsw] Listener error: ${e.message}\n`);
                }
            }
        }

        // Wait for async response
        await new Promise(r => setTimeout(r, 5000));
        if (!resolved) {
            clearTimeout(timeout);
            reject(new Error('HSW listeners did not respond within 5s'));
        }
    });
}

function createXHRClass() {
    return class XMLHttpRequest {
        constructor() {
            this.readyState = 0;
            this._method = ''; this._url = ''; this._headers = {};
            this.status = 0; this.statusText = '';
            this.responseText = ''; this.response = '';
            this.responseType = ''; this.responseURL = '';
            this.withCredentials = false; this.timeout = 0;
            this.onreadystatechange = null; this.onload = null;
            this.onerror = null; this.onprogress = null;
            this.onloadstart = null; this.onloadend = null;
            this.ontimeout = null; this.onabort = null;
            this._listeners = {};
        }
        open(method, url, async) { this._method = method; this._url = url; this.readyState = 1; this._fireRSC(); }
        setRequestHeader(key, value) { this._headers[key] = value; }
        send(body) {
            const self = this;
            makeRequest(this._url, { method: this._method, headers: this._headers, body })
                .then(resp => {
                    self.status = resp.status;
                    self.statusText = resp.status === 200 ? 'OK' : String(resp.status);
                    self.responseText = resp.body;
                    self.responseURL = resp.url;
                    if (self.responseType === 'json') {
                        try { self.response = JSON.parse(resp.body); } catch { self.response = null; }
                    } else if (self.responseType === 'arraybuffer') {
                        const buf = Buffer.from(resp.body, 'binary');
                        self.response = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                    } else {
                        self.response = resp.body;
                    }
                    self.readyState = 2; self._fireRSC();
                    self.readyState = 3; self._fireRSC();
                    self.readyState = 4; self._fireRSC();
                    self._fire('load');
                    self._fire('loadend');
                })
                .catch(e => { self._fire('error', e); self._fire('loadend'); });
        }
        abort() { this._fire('abort'); }
        getResponseHeader(k) { return null; }
        getAllResponseHeaders() { return ''; }
        overrideMimeType() {}
        addEventListener(t, fn) {
            if (!this._listeners[t]) this._listeners[t] = [];
            this._listeners[t].push(fn);
        }
        removeEventListener(t, fn) {
            if (this._listeners[t]) this._listeners[t] = this._listeners[t].filter(f => f !== fn);
        }
        _fire(type, arg) {
            const evt = { type, target: this, currentTarget: this };
            const prop = 'on' + type;
            if (typeof this[prop] === 'function') this[prop](evt);
            if (this._listeners[type]) this._listeners[type].forEach(fn => fn(evt));
        }
        _fireRSC() { this._fire('readystatechange'); }
        static UNSENT = 0; static OPENED = 1; static HEADERS_RECEIVED = 2; static LOADING = 3; static DONE = 4;
    };
}

async function tryDirectCall(ctx, sandbox, req) {
    const me = sandbox.module?.exports;
    if (typeof me === 'function') { try { return await me(req); } catch {} }
    if (me && typeof me === 'object') {
        for (const key of Object.keys(me)) {
            if (typeof me[key] === 'function') {
                try { const r = await me[key](req); if (typeof r === 'string' && r.length > 10) return r; } catch {}
            }
        }
    }
    return null;
}


// Replace the entire solve() function:

async function solve(config) {
    const { sitekey, host, version } = config;
    const hswPath = path.join(__dirname, 'hsw.js');
    const output = { steps: [], token: null, error: null };

    function logStep(name, data) {
        output.steps.push({ name, ...data });
        process.stderr.write(`  [${name}] ${JSON.stringify(data).substring(0, 200)}\n`);
    }

    // Step 1: checksiteconfig
    logStep('checksiteconfig', { status: 'starting' });
    const configResp = await makeRequest(
        `https://api.hcaptcha.com/checksiteconfig?v=${version}&host=${host}&sitekey=${sitekey}&sc=1&swa=1&spst=1`,
        {
            method: 'POST',
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'Origin': 'https://newassets.hcaptcha.com',
                'Referer': `https://newassets.hcaptcha.com/captcha/v1/${version}/static/hcaptcha.html`,
                'Content-Length': '0',
            },
        }
    );
    const siteConfig = JSON.parse(configResp.body);
    logStep('checksiteconfig', { pass: siteConfig.pass, features: siteConfig.features, cType: siteConfig.c?.type });
    const cData = siteConfig.c;
    if (!cData || !cData.req) { output.error = 'No challenge data'; return output; }

    // Step 2: HSW proof
    if (!fs.existsSync(hswPath)) { output.error = 'hsw.js not found'; return output; }
    logStep('hsw', { status: 'solving' });
    const hswCode = fs.readFileSync(hswPath, 'utf-8');

    let proof;
    try { proof = await solveHSW(hswCode, cData.req); } catch (e) {
        logStep('hsw', { status: 'error', error: e.message });
        output.error = `HSW failed: ${e.message}`;
        return output;
    }
    logStep('hsw', { status: 'done', proofLen: proof?.length });
    if (!proof || proof.length < 5) { output.error = 'HSW returned empty proof'; return output; }

    // Step 3: Load the hcaptcha.html inline bundle and run it in a sandbox
    // to make the getcaptcha request with proper enc_get_req encryption
    logStep('widget', { status: 'loading_bundle' });

    const assetBase = `https://newassets.hcaptcha.com/captcha/v1/${version}/static`;
    const htmlResp = await makeRequest(`${assetBase}/hcaptcha.html`, {
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        },
    });

    // Extract inline scripts
    const scriptPattern = /<script[^>]*>([\s\S]*?)<\/script>/gi;
    const scripts = [];
    let sm;
    while ((sm = scriptPattern.exec(htmlResp.body)) !== null) {
        if (sm[1].trim().length > 100) scripts.push(sm[1].trim());
    }
    logStep('widget', { scripts: scripts.length, sizes: scripts.map(s => s.length) });

    // Find the main bundle (largest script with getcaptcha)
    let mainScript = null;
    for (const s of scripts) {
        if (s.includes('getcaptcha')) {
            mainScript = s;
            break;
        }
    }
    if (!mainScript) {
        // Use the largest script
        mainScript = scripts.sort((a, b) => b.length - a.length)[0];
    }

    if (!mainScript) {
        output.error = 'No inline script found in hcaptcha.html';
        return output;
    }

    logStep('widget', { mainScriptSize: mainScript.length, hasGetcaptcha: mainScript.includes('getcaptcha'), hasEncReq: mainScript.includes('enc_get_req') });

    // Step 4: Run the widget bundle in a sandbox and intercept the getcaptcha request
    logStep('widget', { status: 'running_sandbox' });

    // Build motion data
    const now = Date.now();
    const st = now - 3000 - Math.floor(Math.random() * 5000);
    const mm = [];
    let t = st + 200 + Math.floor(Math.random() * 300);
    let mx = 150 + Math.floor(Math.random() * 200);
    let my = 200 + Math.floor(Math.random() * 200);
    for (let i = 0; i < 25 + Math.floor(Math.random() * 30); i++) {
        mx = Math.max(10, Math.min(790, mx + Math.floor(Math.random() * 41) - 20));
        my = Math.max(10, Math.min(590, my + Math.floor(Math.random() * 41) - 20));
        t += 8 + Math.floor(Math.random() * 42);
        mm.push([mx, my, t]);
    }
    const md = [[mx, my, t + 30 + Math.floor(Math.random() * 70)]];
    const mu = [[mx + Math.floor(Math.random() * 5) - 2, my + Math.floor(Math.random() * 5) - 2, md[0][2] + 60 + Math.floor(Math.random() * 90)]];

    const motionData = {
        st, dct: st + 100 + Math.floor(Math.random() * 400),
        mm, md, mu, km: [], kd: [], ku: [],
        topLevel: {
            st: st - 2000 - Math.floor(Math.random() * 3000),
            sc: { availWidth: 1920, availHeight: 1040, width: 1920, height: 1080, colorDepth: 24, pixelDepth: 24, availLeft: 0, availTop: 0 },
            nv: { hardwareConcurrency: 8, deviceMemory: 8, maxTouchPoints: 0 },
            dr: "", inv: false, exec: false,
        },
        v: 1,
    };

    // Create the sandbox and run the widget
    const widgetResult = await runWidgetBundle(mainScript, {
        sitekey, host, version, proof, cData, motionData, logStep,
    });

    if (widgetResult.captchaResponse) {
        let directResult;
        try { directResult = JSON.parse(widgetResult.captchaResponse.body); } catch { directResult = { raw: widgetResult.captchaResponse.body.substring(0, 500) }; }

        logStep('result', {
            status: widgetResult.captchaResponse.status,
            success: directResult.success,
            errorCodes: directResult['error-codes'],
            hasTasklist: !!directResult.tasklist,
            taskCount: directResult.tasklist?.length,
            hasToken: !!directResult.generated_pass_UUID,
            requestType: directResult.request_type,
            keys: Object.keys(directResult),
        });

        if (directResult.tasklist) {
            output.challenge = {
                key: directResult.key,
                type: directResult.request_type,
                question: directResult.requester_question,
                taskCount: directResult.tasklist.length,
                tasks: directResult.tasklist.map(t => ({ key: t.task_key, url: t.datapoint_uri })),
                c: directResult.c,
            };
        }
        if (directResult.generated_pass_UUID) output.token = directResult.generated_pass_UUID;
        output.directResult = directResult;
    } else {
        logStep('widget', { status: 'no_intercept', error: widgetResult.error });

        // Fallback: try plain request
        logStep('fallback', { status: 'plain_request' });

        const formBody = new URLSearchParams();
        formBody.append('v', version);
        formBody.append('sitekey', sitekey);
        formBody.append('host', host);
        formBody.append('hl', 'en');
        formBody.append('motionData', JSON.stringify(motionData));
        formBody.append('n', proof);
        formBody.append('c', JSON.stringify(cData));
        const bodyStr = formBody.toString();

        const directResp = await makeRequest(`https://api.hcaptcha.com/getcaptcha/${sitekey}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Content-Length': Buffer.byteLength(bodyStr).toString(),
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': '*/*',
                'Origin': 'https://newassets.hcaptcha.com',
                'Referer': `https://newassets.hcaptcha.com/captcha/v1/${version}/static/hcaptcha.html`,
            },
            body: bodyStr,
        });

        let directResult;
        try { directResult = JSON.parse(directResp.body); } catch { directResult = { raw: directResp.body.substring(0, 500) }; }

        logStep('fallback_result', {
            status: directResp.status, success: directResult.success,
            errorCodes: directResult['error-codes'],
            hasTasklist: !!directResult.tasklist,
            keys: Object.keys(directResult),
        });

        if (directResult.tasklist) {
            output.challenge = {
                key: directResult.key, type: directResult.request_type, question: directResult.requester_question,
                taskCount: directResult.tasklist.length,
                tasks: directResult.tasklist.map(t => ({ key: t.task_key, url: t.datapoint_uri })),
                c: directResult.c,
            };
        }
        if (directResult.generated_pass_UUID) output.token = directResult.generated_pass_UUID;
        output.directResult = directResult;
    }

    return output;
}

async function runWidgetBundle(scriptCode, opts) {
    const { sitekey, host, version, proof, cData, motionData, logStep } = opts;

    return new Promise(async (resolve) => {
        const timeout = setTimeout(() => {
            resolve({ captchaResponse: null, error: 'Widget sandbox timed out' });
        }, 30000);

        let captchaResponse = null;
        let checksiteIntercepted = false;

        const frameId = 'oauthgqf1yzp';
        const widgetId = `${Date.now().toString(36)}`;

        const doc = createDocument();
        const locationObj = {
            href: `https://newassets.hcaptcha.com/captcha/v1/${version}/static/hcaptcha.html#frame=checkbox&id=${frameId}&host=${host}&sitekey=${sitekey}&size=normal&theme=light&origin=https%3A%2F%2F${host}`,
            origin: 'https://newassets.hcaptcha.com',
            protocol: 'https:',
            host: 'newassets.hcaptcha.com',
            hostname: 'newassets.hcaptcha.com',
            pathname: `/captcha/v1/${version}/static/hcaptcha.html`,
            port: '',
            search: '',
            hash: `#frame=checkbox&id=${frameId}&host=${host}&sitekey=${sitekey}&size=normal&theme=light&origin=https%3A%2F%2F${host}`,
            assign() {}, reload() {}, replace() {},
            toString() { return this.href; },
        };
        doc.location = locationObj;

        // Intercept network requests
        const interceptFetch = async (url, fetchOpts) => {
            const urlStr = typeof url === 'string' ? url : url?.url || String(url);

            logStep('net', { url: urlStr.substring(0, 80), method: fetchOpts?.method || 'GET' });

            // Intercept checksiteconfig - return our already-fetched data
            if (urlStr.includes('checksiteconfig')) {
                checksiteIntercepted = true;
                const configJson = JSON.stringify({
                    pass: true, success: true, c: cData,
                    features: { enc_get_req: true },
                });
                return {
                    ok: true, status: 200,
                    headers: { get() { return 'application/json'; }, has() { return false; }, forEach() {} },
                    json: async () => JSON.parse(configJson),
                    text: async () => configJson,
                    clone() { return this; },
                };
            }

            // Intercept getcaptcha - this is what we need!
            if (urlStr.includes('getcaptcha')) {
                logStep('intercepted_getcaptcha', {
                    method: fetchOpts?.method,
                    contentType: fetchOpts?.headers?.['Content-Type'] || fetchOpts?.headers?.['content-type'],
                    bodyLen: fetchOpts?.body?.length,
                    bodyPreview: typeof fetchOpts?.body === 'string' ? fetchOpts.body.substring(0, 300) : undefined,
                });

                // Forward the actual encrypted request to hcaptcha
                const resp = await makeRequest(urlStr, {
                    method: fetchOpts?.method || 'POST',
                    headers: fetchOpts?.headers || {},
                    body: fetchOpts?.body,
                });

                captchaResponse = resp;
                clearTimeout(timeout);

                const bodyText = resp.body;
                const result = {
                    ok: resp.status >= 200 && resp.status < 300,
                    status: resp.status,
                    headers: {
                        get: (k) => resp.headers[k.toLowerCase()] || null,
                        has: (k) => k.toLowerCase() in resp.headers,
                        forEach: (cb) => Object.entries(resp.headers).forEach(([k, v]) => cb(v, k)),
                    },
                    json: async () => JSON.parse(bodyText),
                    text: async () => bodyText,
                    arrayBuffer: async () => Buffer.from(bodyText, 'binary').buffer,
                    clone() { return this; },
                };

                // Resolve our promise with the response
                setTimeout(() => resolve({ captchaResponse: resp }), 100);
                return result;
            }

            // HSW request - return our proof
            if (urlStr.includes('/hsw')) {
                // The widget fetches hsw.js from the URL in the JWT
                const resp = await makeRequest(urlStr, {
                    method: fetchOpts?.method || 'GET',
                    headers: fetchOpts?.headers || {},
                    body: fetchOpts?.body,
                });
                const bodyText = resp.body;
                return {
                    ok: resp.status >= 200 && resp.status < 300,
                    status: resp.status,
                    headers: {
                        get: (k) => resp.headers[k.toLowerCase()] || null,
                        has: (k) => k.toLowerCase() in resp.headers,
                        forEach: (cb) => Object.entries(resp.headers).forEach(([k, v]) => cb(v, k)),
                    },
                    json: async () => JSON.parse(bodyText),
                    text: async () => bodyText,
                    arrayBuffer: async () => {
                        const buf = Buffer.from(bodyText, 'binary');
                        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                    },
                    clone() { return this; },
                };
            }

            // All other requests - pass through
            try {
                const resp = await makeRequest(urlStr, {
                    method: fetchOpts?.method || 'GET',
                    headers: fetchOpts?.headers || {},
                    body: fetchOpts?.body,
                });
                const bodyText = resp.body;
                return {
                    ok: resp.status >= 200 && resp.status < 300,
                    status: resp.status,
                    headers: {
                        get: (k) => resp.headers[k.toLowerCase()] || null,
                        has: (k) => k.toLowerCase() in resp.headers,
                        forEach: (cb) => Object.entries(resp.headers).forEach(([k, v]) => cb(v, k)),
                    },
                    json: async () => JSON.parse(bodyText),
                    text: async () => bodyText,
                    arrayBuffer: async () => {
                        const buf = Buffer.from(bodyText, 'binary');
                        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                    },
                    clone() { return this; },
                };
            } catch (e) {
                return { ok: false, status: 0, headers: { get() { return null; }, has() { return false; }, forEach() {} }, json: async () => ({}), text: async () => '', clone() { return this; } };
            }
        };

        // Message handling - the widget communicates via postMessage
        const parentMessages = [];
        const messageListeners = [];

        const contextObj = {
            document: doc,
            location: locationObj,
            navigator: {
                userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                language: 'en-US', languages: ['en-US', 'en'], platform: 'Win32',
                hardwareConcurrency: 8, deviceMemory: 8, maxTouchPoints: 0,
                onLine: true, vendor: 'Google Inc.', webdriver: false, cookieEnabled: true,
                plugins: { length: 0, item() { return null; }, namedItem() { return null; }, refresh() {} },
                mimeTypes: { length: 0, item() { return null; }, namedItem() { return null; } },
                doNotTrack: null,
                getBattery: async () => ({ charging: true, chargingTime: 0, dischargingTime: Infinity, level: 1, addEventListener() {} }),
                connection: { effectiveType: '4g', rtt: 50, downlink: 10, saveData: false, addEventListener() {} },
                mediaDevices: { enumerateDevices: async () => [], addEventListener() {} },
                permissions: { query: async () => ({ state: 'prompt', addEventListener() {} }) },
                clipboard: { readText: async () => '', writeText: async () => {} },
                locks: { request: async (name, cb) => { if (cb) return cb({ mode: 'exclusive', name }); } },
                storage: { estimate: async () => ({ quota: 0, usage: 0 }), persist: async () => false },
                serviceWorker: { ready: new Promise(() => {}), register: async () => ({}), addEventListener() {}, controller: null },
                sendBeacon() { return true; },
                userAgentData: {
                    brands: [{ brand: 'Chromium', version: '131' }, { brand: 'Not_A Brand', version: '24' }],
                    mobile: false, platform: 'Windows',
                    getHighEntropyValues: async () => ({
                        architecture: 'x86', bitness: '64', model: '', platformVersion: '15.0.0',
                        fullVersionList: [{ brand: 'Chromium', version: '131.0.6778.86' }],
                    }),
                },
            },
            screen: {
                width: 1920, height: 1080, availWidth: 1920, availHeight: 1040,
                colorDepth: 24, pixelDepth: 24, availLeft: 0, availTop: 0,
                orientation: { type: 'landscape-primary', angle: 0, addEventListener() {} },
            },
            history: { length: 1, state: null, pushState() {}, replaceState() {}, go() {}, back() {}, forward() {} },
            crypto: webcrypto,
            console: { log() {}, warn() {}, error() {}, info() {}, debug() {}, trace() {}, dir() {}, time() {}, timeEnd() {}, table() {}, group() {}, groupEnd() {}, assert() {} },
            performance: {
                now: (() => { const s = Date.now(); return () => Date.now() - s; })(),
                timeOrigin: Date.now() - 5000,
                timing: { navigationStart: Date.now() - 5000, fetchStart: Date.now() - 4900, domContentLoadedEventEnd: Date.now() - 3000, loadEventEnd: Date.now() - 2500 },
                memory: { jsHeapSizeLimit: 4294705152, totalJSHeapSize: 35000000, usedJSHeapSize: 25000000 },
                getEntries() { return []; }, getEntriesByType() { return []; }, getEntriesByName() { return []; },
                mark() {}, measure() {}, clearMarks() {}, clearMeasures() {},
            },
            localStorage: { _d: {}, getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; }, clear() { this._d = {}; }, get length() { return Object.keys(this._d).length; }, key(i) { return Object.keys(this._d)[i] || null; } },
            sessionStorage: { _d: {}, getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = String(v); }, removeItem(k) { delete this._d[k]; }, clear() { this._d = {}; }, get length() { return Object.keys(this._d).length; }, key(i) { return Object.keys(this._d)[i] || null; } },
            setTimeout, clearTimeout, setInterval, clearInterval, queueMicrotask,
            requestAnimationFrame: (cb) => setTimeout(cb, 16),
            cancelAnimationFrame: clearTimeout,
            requestIdleCallback: (cb) => setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 50 }), 1),
            cancelIdleCallback: clearTimeout,
            atob: (s) => Buffer.from(s, 'base64').toString('binary'),
            btoa: (s) => Buffer.from(s, 'binary').toString('base64'),
            TextEncoder, TextDecoder, URL, URLSearchParams,
            Uint8Array, Int32Array, Float64Array, Float32Array, Uint32Array, Uint16Array,
            Int8Array, Int16Array, BigInt64Array, BigUint64Array, Uint8ClampedArray,
            ArrayBuffer, DataView, SharedArrayBuffer,
            Map, Set, WeakMap, WeakSet, WeakRef, FinalizationRegistry,
            Promise, Proxy, Reflect, Symbol, BigInt, WebAssembly,
            Math, Date, JSON, String, Number, Boolean, Array, Object, RegExp, Function,
            Error, TypeError, RangeError, ReferenceError, SyntaxError, URIError, EvalError, AggregateError,
            parseInt, parseFloat, isNaN, isFinite, NaN, Infinity, undefined,
            encodeURIComponent, decodeURIComponent, encodeURI, decodeURI,
            escape, unescape,
            innerWidth: 1920, innerHeight: 969, outerWidth: 1920, outerHeight: 1040,
            devicePixelRatio: 1, screenX: 0, screenY: 0, screenLeft: 0, screenTop: 0,
            pageXOffset: 0, pageYOffset: 0, scrollX: 0, scrollY: 0,
            visualViewport: { width: 1920, height: 969, offsetTop: 0, offsetLeft: 0, scale: 1, addEventListener() {}, removeEventListener() {} },
            origin: 'https://newassets.hcaptcha.com',
            isSecureContext: true,
            crossOriginIsolated: false,
            fetch: interceptFetch,
            XMLHttpRequest: createInterceptingXHRClass((url, method, headers, body) => {
                logStep('xhr', { url: url.substring(0, 80), method, bodyLen: body?.length });
                if (url.includes('getcaptcha')) {
                    logStep('xhr_getcaptcha', {
                        bodyLen: body?.length,
                        bodyPreview: typeof body === 'string' ? body.substring(0, 300) : undefined,
                    });
                }
            }),
            postMessage: function(data, targetOrigin) {
                parentMessages.push({ data, targetOrigin });
                logStep('postMessage', { dataType: typeof data, preview: JSON.stringify(data).substring(0, 100) });
            },
            addEventListener: function(type, fn) {
                if (type === 'message') messageListeners.push(fn);
            },
            removeEventListener() {},
            close() {},
            Blob: class Blob {
                constructor(parts = [], opts = {}) {
                    this._parts = parts; this.type = opts.type || '';
                    this.size = parts.reduce((s, p) => s + (p?.length || p?.byteLength || 0), 0);
                }
                async text() { return this._parts.map(p => typeof p === 'string' ? p : '').join(''); }
                async arrayBuffer() {
                    const bufs = this._parts.map(p => {
                        if (p instanceof ArrayBuffer) return Buffer.from(p);
                        if (ArrayBuffer.isView(p)) return Buffer.from(p.buffer, p.byteOffset, p.byteLength);
                        return Buffer.from(String(p));
                    });
                    const c = Buffer.concat(bufs);
                    return c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength);
                }
                slice() { return new Blob(); }
            },
            File: class File { constructor(p, n, o) { this.name = n; this.type = o?.type || ''; this.size = 0; } },
            FileReader: class FileReader {
                readAsDataURL() { setTimeout(() => this.onload?.({ target: { result: 'data:;base64,' } }), 0); }
                readAsArrayBuffer() { setTimeout(() => this.onload?.({ target: { result: new ArrayBuffer(0) } }), 0); }
                readAsText() { setTimeout(() => this.onload?.({ target: { result: '' } }), 0); }
                addEventListener(t, fn) { if (t === 'load') this.onload = fn; }
            },
            FormData: class FormData {
                constructor() { this._d = []; }
                append(k, v) { this._d.push([k, v]); }
                get(k) { const e = this._d.find(([kk]) => kk === k); return e ? e[1] : null; }
                getAll(k) { return this._d.filter(([kk]) => kk === k).map(([, v]) => v); }
                has(k) { return this._d.some(([kk]) => kk === k); }
                delete(k) { this._d = this._d.filter(([kk]) => kk !== k); }
                forEach(cb) { this._d.forEach(([k, v]) => cb(v, k)); }
                entries() { return this._d[Symbol.iterator](); }
            },
            Worker: class Worker {
                constructor(url) {
                    this.onmessage = null;
                    this._url = typeof url === 'string' ? url : url?.url || '';
                    this._messageHandler = null;

                    // If this is the HSW worker, we need to handle it
                    if (this._url.includes('hsw') || this._url.includes('/c/')) {
                        logStep('worker', { type: 'hsw_worker', url: this._url.substring(0, 80) });
                    }
                }
                postMessage(data) {
                    // If this worker is for HSW, respond with our precomputed proof
                    if (this._url.includes('hsw') || this._url.includes('/c/') || (data && (data.type === 'hsw' || typeof data === 'string'))) {
                        logStep('worker_msg', { dataType: typeof data, isHSW: true });
                        const self = this;
                        setTimeout(() => {
                            const evt = { data: proof };
                            if (self.onmessage) self.onmessage(evt);
                            if (self._messageHandler) self._messageHandler(evt);
                        }, 50);
                    }
                }
                terminate() {}
                addEventListener(t, fn) {
                    if (t === 'message') {
                        this._messageHandler = fn;
                    }
                }
                removeEventListener() {}
            },
            MessageChannel: class MessageChannel {
                constructor() {
                    this.port1 = { postMessage() {}, addEventListener() {}, start() {}, close() {}, onmessage: null };
                    this.port2 = { postMessage() {}, addEventListener() {}, start() {}, close() {}, onmessage: null };
                }
            },
            BroadcastChannel: class BroadcastChannel { constructor() { this.onmessage = null; } postMessage() {} close() {} addEventListener() {} },
            Event: class Event {
                constructor(t, i) { this.type = t; this.bubbles = i?.bubbles || false; this.cancelable = i?.cancelable || false; this.defaultPrevented = false; this.target = null; this.currentTarget = null; this.isTrusted = true; }
                preventDefault() { this.defaultPrevented = true; } stopPropagation() {} stopImmediatePropagation() {}
            },
            CustomEvent: class CustomEvent { constructor(t, i) { this.type = t; this.detail = i?.detail; } preventDefault() {} stopPropagation() {} },
            MessageEvent: class MessageEvent { constructor(t, i) { this.type = t; this.data = i?.data; this.origin = i?.origin || ''; this.source = i?.source || null; } },
            ErrorEvent: class ErrorEvent { constructor(t, i) { this.type = t; this.message = i?.message; this.error = i?.error; } },
            MutationObserver: class MutationObserver { observe() {} disconnect() {} takeRecords() { return []; } },
            IntersectionObserver: class IntersectionObserver { observe() {} unobserve() {} disconnect() {} },
            ResizeObserver: class ResizeObserver { observe() {} unobserve() {} disconnect() {} },
            PerformanceObserver: class PerformanceObserver { observe() {} disconnect() {} static supportedEntryTypes = [] },
            Image: class Image {
                constructor(w, h) { this.width = w || 0; this.height = h || 0; this.src = ''; this.complete = true; this.naturalWidth = 300; this.naturalHeight = 150; }
                addEventListener() {} removeEventListener() {}
            },
            DOMParser: class DOMParser { parseFromString() { return createDocument(); } },
            AbortController: class AbortController {
                constructor() { this.signal = { aborted: false, reason: undefined, addEventListener() {}, removeEventListener() {}, throwIfAborted() {}, onabort: null }; }
                abort(r) { this.signal.aborted = true; this.signal.reason = r; if (this.signal.onabort) this.signal.onabort(); }
            },
            Headers: class Headers {
                constructor(init) { this._h = {}; if (init) { if (typeof init.forEach === 'function') init.forEach((v, k) => { this._h[k.toLowerCase()] = v; }); else if (typeof init === 'object') Object.entries(init).forEach(([k, v]) => { this._h[k.toLowerCase()] = v; }); } }
                get(k) { return this._h[k.toLowerCase()] || null; } set(k, v) { this._h[k.toLowerCase()] = v; }
                has(k) { return k.toLowerCase() in this._h; } delete(k) { delete this._h[k.toLowerCase()]; }
                forEach(cb) { Object.entries(this._h).forEach(([k, v]) => cb(v, k)); }
                entries() { return Object.entries(this._h); }
                keys() { return Object.keys(this._h); }
                values() { return Object.values(this._h); }
                [Symbol.iterator]() { return Object.entries(this._h)[Symbol.iterator](); }
            },
            Request: class Request { constructor(u, o) { this.url = typeof u === 'string' ? u : u?.url; this.method = o?.method || 'GET'; this.headers = o?.headers || {}; this.body = o?.body; } },
            Response: class Response {
                constructor(b, i) { this._body = b; this.status = i?.status || 200; this.ok = this.status < 400; this.headers = new Map(); }
                async json() { return JSON.parse(this._body); } async text() { return this._body; }
                static json(data, init) { return new Response(JSON.stringify(data), init); }
            },
            Element: class Element {}, HTMLElement: class HTMLElement {},
            HTMLCanvasElement: class HTMLCanvasElement {}, HTMLImageElement: class HTMLImageElement {},
            HTMLVideoElement: class HTMLVideoElement {}, HTMLScriptElement: class HTMLScriptElement {},
            HTMLIFrameElement: class HTMLIFrameElement {}, Node: class Node {}, NodeList: class NodeList {},
            DOMRect: class DOMRect { constructor(x, y, w, h) { this.x = x || 0; this.y = y || 0; this.width = w || 0; this.height = h || 0; } },
            DOMRectReadOnly: class DOMRectReadOnly { constructor(x, y, w, h) { this.x = x || 0; this.y = y || 0; this.width = w || 0; this.height = h || 0; } },
            CSS: { supports() { return false; }, escape(s) { return s; } },
            CSSStyleDeclaration: class CSSStyleDeclaration {},
            matchMedia: () => ({ matches: false, media: '', addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} }),
            getComputedStyle: () => new Proxy({}, { get: (t, p) => typeof p === 'string' ? '' : undefined }),
            getSelection: () => ({ rangeCount: 0, addRange() {}, removeAllRanges() {}, toString() { return ''; } }),
            scrollTo() {}, scrollBy() {}, scroll() {}, focus() {}, blur() {},
            open() { return null; }, print() {}, stop() {},
            alert() {}, confirm() { return false; }, prompt() { return null; },
            indexedDB: {
                open(name) {
                    const req = { result: null, error: null, onsuccess: null, onerror: null, onupgradeneeded: null, addEventListener() {}, readyState: 'done' };
                    setTimeout(() => {
                        req.result = {
                            objectStoreNames: { length: 0, contains() { return false; } },
                            createObjectStore() { return { createIndex() {}, put() { return { onsuccess: null, onerror: null }; } }; },
                            transaction() { return { objectStore() { return { get() { return { onsuccess: null, onerror: null, result: undefined }; }, put() { return { onsuccess: null, onerror: null }; } }; }, oncomplete: null, onerror: null }; },
                            close() {},
                        };
                        if (req.onupgradeneeded) req.onupgradeneeded({ target: req });
                        if (req.onsuccess) req.onsuccess({ target: req });
                    }, 10);
                    return req;
                },
                databases: async () => [],
            },
            Notification: class Notification { static permission = 'denied'; static requestPermission() { return Promise.resolve('denied'); } },
            AudioContext: class AudioContext { constructor() { this.destination = {}; this.sampleRate = 44100; this.state = 'suspended'; } close() { return Promise.resolve(); } createOscillator() { return { connect() {}, start() {}, stop() {}, frequency: { value: 0 } }; } createDynamicsCompressor() { return { connect() {}, threshold: { value: 0 }, knee: { value: 0 }, ratio: { value: 0 }, reduction: { value: 0 }, attack: { value: 0 }, release: { value: 0 } }; } createGain() { return { connect() {}, gain: { value: 1 } }; } createAnalyser() { return { connect() {}, fftSize: 0, getFloatFrequencyData() {} }; } resume() { return Promise.resolve(); } },
            webkitAudioContext: undefined,
            OfflineAudioContext: class OfflineAudioContext { constructor() { this.destination = {}; } startRendering() { return Promise.resolve({ getChannelData() { return new Float32Array(0); } }); } createOscillator() { return { connect() {}, start() {}, stop() {}, frequency: { value: 0 } }; } createDynamicsCompressor() { return { connect() {}, threshold: { value: 0 }, knee: { value: 0 }, ratio: { value: 0 }, attack: { value: 0 }, release: { value: 0 } }; } },
            SpeechSynthesisUtterance: class SpeechSynthesisUtterance {},
            speechSynthesis: { getVoices() { return []; }, addEventListener() {}, speak() {} },
            module: { exports: {} },
            exports: {},
        };

        Object.defineProperty(contextObj, 'onmessage', {
            value: null, writable: true, configurable: true, enumerable: true,
        });

        const ctx = vm.createContext(contextObj, {
            codeGeneration: { strings: true, wasm: true },
        });

        vm.runInContext(`
            self = this;
            window = this;
            globalThis = this;
            top = this;
            parent = { postMessage: postMessage };
            frames = this;
            document.defaultView = this;
        `, ctx);

        try {
            const script = new vm.Script(scriptCode, { filename: 'hcaptcha-bundle.js', timeout: 30000 });
            script.runInContext(ctx);
            logStep('widget', { status: 'script_loaded' });
        } catch (e) {
            logStep('widget', { status: 'script_error', error: e.message.substring(0, 200) });
            clearTimeout(timeout);
            resolve({ captchaResponse: null, error: `Script error: ${e.message}` });
            return;
        }

        // After the script loads, it should initialize and listen for messages
        // We need to simulate the parent frame sending a setup message
        await new Promise(r => setTimeout(r, 500));

        // Send initialization message that the parent frame would send
        logStep('widget', { status: 'sending_init', listeners: messageListeners.length });

        // The widget expects a message from the parent with config
        const initMsg = {
            source: 'hcaptcha',
            label: 'challenge-open',
            id: frameId,
            contents: {
                sitekey: sitekey,
                host: host,
                motionData: JSON.stringify(motionData),
            },
        };

        for (const fn of messageListeners) {
            try {
                fn({
                    data: JSON.stringify(initMsg),
                    origin: `https://${host}`,
                    source: { postMessage() {} },
                });
            } catch (e) {
                logStep('widget', { initError: e.message.substring(0, 100) });
            }
        }

        // Also try simulating a checkbox click
        await new Promise(r => setTimeout(r, 1000));

        const clickMsg = {
            source: 'hcaptcha',
            label: 'checkbox-selected',
            id: frameId,
            contents: { action: 'check' },
        };

        for (const fn of messageListeners) {
            try {
                fn({
                    data: JSON.stringify(clickMsg),
                    origin: `https://${host}`,
                    source: { postMessage() {} },
                });
            } catch (e) {}
        }

        // Wait for the getcaptcha request to be intercepted
        await new Promise(r => setTimeout(r, 8000));

        if (!captchaResponse) {
            logStep('widget', { status: 'no_getcaptcha_intercepted', messagesReceived: parentMessages.length });
            if (parentMessages.length > 0) {
                logStep('widget', { firstMsg: JSON.stringify(parentMessages[0]).substring(0, 200) });
            }
            clearTimeout(timeout);
            resolve({ captchaResponse: null, error: 'getcaptcha was not triggered' });
        }
    });
}


function createInterceptingXHRClass(interceptor) {
    return class XMLHttpRequest {
        constructor() {
            this.readyState = 0;
            this._method = ''; this._url = ''; this._headers = {};
            this.status = 0; this.statusText = '';
            this.responseText = ''; this.response = '';
            this.responseType = ''; this.responseURL = '';
            this.withCredentials = false; this.timeout = 0;
            this.onreadystatechange = null; this.onload = null;
            this.onerror = null; this.onprogress = null;
            this.onloadstart = null; this.onloadend = null;
            this._listeners = {};
        }
        open(method, url) { this._method = method; this._url = url; this.readyState = 1; this._fireRSC(); }
        setRequestHeader(key, value) { this._headers[key] = value; }
        send(body) {
            if (interceptor) interceptor(this._url, this._method, { ...this._headers }, body);
            const self = this;
            makeRequest(this._url, { method: this._method, headers: this._headers, body })
                .then(resp => {
                    self.status = resp.status;
                    self.statusText = resp.status === 200 ? 'OK' : String(resp.status);
                    self.responseText = resp.body;
                    self.responseURL = resp.url;
                    if (self.responseType === 'json') {
                        try { self.response = JSON.parse(resp.body); } catch { self.response = null; }
                    } else if (self.responseType === 'arraybuffer') {
                        const buf = Buffer.from(resp.body, 'binary');
                        self.response = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
                    } else {
                        self.response = resp.body;
                    }
                    self.readyState = 2; self._fireRSC();
                    self.readyState = 3; self._fireRSC();
                    self.readyState = 4; self._fireRSC();
                    self._fire('load');
                    self._fire('loadend');
                })
                .catch(e => { self._fire('error', e); self._fire('loadend'); });
        }
        abort() { this._fire('abort'); }
        getResponseHeader(k) { return null; }
        getAllResponseHeaders() { return ''; }
        overrideMimeType() {}
        addEventListener(t, fn) { if (!this._listeners[t]) this._listeners[t] = []; this._listeners[t].push(fn); }
        removeEventListener(t, fn) { if (this._listeners[t]) this._listeners[t] = this._listeners[t].filter(f => f !== fn); }
        _fire(type) {
            const evt = { type, target: this, currentTarget: this };
            const prop = 'on' + type;
            if (typeof this[prop] === 'function') this[prop](evt);
            if (this._listeners[type]) this._listeners[type].forEach(fn => fn(evt));
        }
        _fireRSC() { this._fire('readystatechange'); }
        static UNSENT = 0; static OPENED = 1; static HEADERS_RECEIVED = 2; static LOADING = 3; static DONE = 4;
    };
}
