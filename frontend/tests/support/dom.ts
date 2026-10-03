import React from 'react'
import { createRoot, type Root } from 'react-dom/client'

export class TestEvent {
  bubbles: boolean
  cancelable: boolean
  defaultPrevented = false
  target: unknown = null
  currentTarget: unknown = null

  constructor(public type: string, init: { bubbles?: boolean; cancelable?: boolean } = {}) {
    this.bubbles = init.bubbles ?? true
    this.cancelable = init.cancelable ?? true
  }

  preventDefault() {
    if (this.cancelable) this.defaultPrevented = true
  }
}

export class TestNode {
  parentNode: TestElement | TestNode | null = null
  childNodes: TestNode[] = []

  constructor(public nodeType: number, public nodeName: string, public ownerDocument: TestDocument) {}

  get firstChild(): TestNode | null {
    return this.childNodes[0] ?? null
  }

  get lastChild(): TestNode | null {
    return this.childNodes.at(-1) ?? null
  }

  appendChild<T extends TestNode>(child: T): T {
    child.parentNode?.removeChild(child as never)
    child.parentNode = this
    this.childNodes.push(child)
    return child
  }

  insertBefore<T extends TestNode>(child: T, before: TestNode | null): T {
    child.parentNode?.removeChild(child as never)
    child.parentNode = this
    const index = before ? this.childNodes.indexOf(before) : -1
    this.childNodes.splice(index < 0 ? this.childNodes.length : index, 0, child)
    return child
  }

  removeChild<T extends TestNode>(child: T): T {
    this.childNodes.splice(this.childNodes.indexOf(child), 1)
    child.parentNode = null
    return child
  }

  addEventListener(_type: string, _listener: (event: TestEvent) => void) {}
  removeEventListener(_type: string, _listener: (event: TestEvent) => void) {}

  dispatchEvent(_event: TestEvent) {
    return true
  }

  get textContent(): string {
    return this.childNodes.map((child) => child.textContent).join('')
  }

  set textContent(value: string | null) {
    this.childNodes = []
    if (value) this.appendChild(this.ownerDocument.createTextNode(value))
  }
}

export class TestText extends TestNode {
  constructor(public data: string, ownerDocument: TestDocument) {
    super(3, '#text', ownerDocument)
  }

  get nodeValue() {
    return this.data
  }

  set nodeValue(value: string) {
    this.data = value
  }

  override get textContent(): string {
    return this.data
  }

  override set textContent(value: string | null) {
    this.data = value ?? ''
  }
}

export class TestElement extends TestNode {
  namespaceURI = 'http://www.w3.org/1999/xhtml'
  style = Object.assign({} as Record<string, string>, {
    setProperty(this: Record<string, unknown>, name: string, value: string) {
      this[name] = value
    },
    removeProperty(this: Record<string, unknown>, name: string) {
      delete this[name]
    },
  })
  attributes = new Map<string, string>()
  listeners = new Map<string, Array<(event: TestEvent) => void>>()
  value = ''
  private defaultValueState = ''
  checked = false
  disabled = false
  multiple = false
  selected = false
  defaultSelected = false

  constructor(public tagName: string, ownerDocument: TestDocument) {
    super(1, tagName.toUpperCase(), ownerDocument)
    this.tagName = tagName.toUpperCase()
  }

  get type() {
    return this.attributes.get('type') ?? ''
  }

  set type(value: string) {
    this.setAttribute('type', value)
  }

  get name() {
    return this.attributes.get('name') ?? ''
  }

  set name(value: string) {
    this.setAttribute('name', value)
  }

  get id() {
    return this.attributes.get('id') ?? ''
  }

  set id(value: string) {
    this.setAttribute('id', value)
  }

  get htmlFor() {
    return this.attributes.get('for') ?? ''
  }

  set htmlFor(value: string) {
    this.setAttribute('for', value)
  }

  get defaultValue() {
    return this.defaultValueState
  }

  set defaultValue(value: string) {
    this.defaultValueState = value
    if (this.tagName === 'TEXTAREA') this.value = value
  }

  get options(): TestElement[] {
    return this.childNodes.flatMap((child) => (
      child instanceof TestElement
        ? child.tagName === 'OPTION' ? [child] : child.options
        : []
    ))
  }

  override get textContent(): string {
    return super.textContent
  }

  override set textContent(value: string | null) {
    super.textContent = value
    if (this.tagName === 'TEXTAREA') this.value = value ?? ''
  }

  setAttribute(name: string, value: string) {
    this.attributes.set(name, String(value))
  }

  getAttribute(name: string) {
    return this.attributes.get(name) ?? null
  }

  getAttributeNames() {
    return [...this.attributes.keys()]
  }

  removeAttribute(name: string) {
    this.attributes.delete(name)
  }

  setAttributeNS(_namespace: string | null, name: string, value: string) {
    this.setAttribute(name, value)
  }

  removeAttributeNS(_namespace: string | null, name: string) {
    this.removeAttribute(name)
  }

  override addEventListener(type: string, listener: (event: TestEvent) => void) {
    const listeners = this.listeners.get(type) ?? []
    listeners.push(listener)
    this.listeners.set(type, listeners)
  }

  override removeEventListener(type: string, listener: (event: TestEvent) => void) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((item) => item !== listener))
  }

  override dispatchEvent(event: TestEvent) {
    if (!event.target) event.target = this
    event.currentTarget = this
    for (const listener of this.listeners.get(event.type) ?? []) listener(event)
    if (event.bubbles && this.parentNode) this.parentNode.dispatchEvent(event)
    return !event.defaultPrevented
  }

  focus() {
    this.ownerDocument.activeElement = this
  }

  click() {
    if (this.tagName === 'INPUT' && this.type === 'radio') this.checked = true
    const event = new TestEvent('click')
    this.dispatchEvent(event)
    if (this.tagName === 'INPUT') {
      this.dispatchEvent(new TestEvent('input'))
      this.dispatchEvent(new TestEvent('change'))
    }
    if (!event.defaultPrevented && this.tagName === 'BUTTON' && (this.type === '' || this.type === 'submit')) {
      this.closest('form')?.dispatchEvent(new TestEvent('submit'))
    }
  }

  closest(tag: string) {
    let node: TestElement | TestNode | null = this
    const wanted = tag.toUpperCase()
    while (node) {
      if (node instanceof TestElement && node.tagName === wanted) return node
      node = node.parentNode
    }
    return null
  }

  querySelectorAll(selector: string) {
    const radioMatch = selector.match(/^input\[name="([^"]+)"\]\[type="radio"\]$/)
    if (!radioMatch) return []
    return descendants(this).filter((node) =>
      node.tagName === 'INPUT' && node.name === radioMatch[1] && node.type === 'radio')
  }
}

export class TestDocument extends TestNode {
  defaultView: Record<string, unknown>
  documentElement: TestElement
  body: TestElement
  activeElement: TestElement | null = null

  constructor() {
    super(9, '#document', undefined as unknown as TestDocument)
    this.ownerDocument = this
    this.documentElement = this.createElement('html')
    this.body = this.createElement('body')
    this.appendChild(this.documentElement)
    this.documentElement.appendChild(this.body)
    this.defaultView = {
      ...globalThis,
      document: this,
      Node: TestNode,
      Element: TestElement,
      HTMLElement: TestElement,
      HTMLInputElement: TestElement,
      HTMLIFrameElement: TestElement,
      HTMLTextAreaElement: TestElement,
      HTMLButtonElement: TestElement,
      event: undefined,
      getSelection: () => null,
    }
  }

  createElement(tagName: string) {
    return new TestElement(tagName, this)
  }

  createElementNS(_namespace: string, tagName: string) {
    return this.createElement(tagName)
  }

  createTextNode(data: string) {
    return new TestText(data, this)
  }

  createComment(data: string) {
    return new TestText(data, this)
  }
}

export function descendants(node: TestNode): TestElement[] {
  return node.childNodes.flatMap((child) => child instanceof TestElement
    ? [child, ...descendants(child)]
    : [])
}

export function textOf(node: TestNode) {
  // Strip the zero-width space GoalAnnouncerProvider toggles on repeat announcements
  // (forces a DOM text change for screen readers) — it's not visible/meaningful text.
  return node.textContent.replace(/​/g, '').replace(/\s+/g, ' ').trim()
}

export function findElement(node: TestNode, tagName: string, attribute?: [string, string]): TestElement | undefined {
  if (
    node instanceof TestElement
    && node.tagName === tagName.toUpperCase()
    && (!attribute || node.getAttribute(attribute[0]) === attribute[1])
  ) return node
  for (const child of node.childNodes) {
    const match = findElement(child, tagName, attribute)
    if (match) return match
  }
}

export function getButton(root: TestNode, label: string) {
  const button = descendants(root).find((node) => node.tagName === 'BUTTON' && textOf(node) === label)
  if (!button) throw new Error(`Button not found: ${label}`)
  return button
}

export function getByLabel(root: TestNode, label: string, tagName?: string) {
  const labels = descendants(root).filter((node) => node.tagName === 'LABEL')
  const labelNode = labels.find((node) => textOf(node).includes(label))
  if (!labelNode) throw new Error(`Label not found: ${label}`)
  const controls = descendants(labelNode).filter((node) => !tagName || node.tagName === tagName.toUpperCase())
  const linkedControl = descendants(root).find((node) =>
    labelNode.htmlFor && node.id === labelNode.htmlFor && (!tagName || node.tagName === tagName.toUpperCase()))
  const control = controls[0] ?? linkedControl
  if (!control) throw new Error(`Control not found for: ${label}`)
  return control
}

export function changeControl(control: TestElement, value: string | boolean) {
  if (typeof value === 'boolean') control.checked = value
  else control.value = value
  const propsKey = Object.keys(control).find((key) => key.startsWith('__reactProps$'))
  const onChange = propsKey
    ? (control as unknown as Record<string, { onChange?: (event: { target: TestElement }) => void }>)[propsKey]?.onChange
    : null
  onChange?.({ target: control })
}

export async function mountWithTestDom(
  render: () => React.ReactNode,
  act: (callback: () => void | Promise<void>) => Promise<void> | void,
) {
  const setGlobal = (name: string, value: unknown) => {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      writable: true,
      value,
    })
  }
  const previousGlobals = {
    document: globalThis.document,
    window: globalThis.window,
    Event: globalThis.Event,
    Node: globalThis.Node,
    Element: globalThis.Element,
    HTMLElement: globalThis.HTMLElement,
    HTMLInputElement: globalThis.HTMLInputElement,
    HTMLIFrameElement: globalThis.HTMLIFrameElement,
    HTMLTextAreaElement: globalThis.HTMLTextAreaElement,
    HTMLButtonElement: globalThis.HTMLButtonElement,
    navigator: globalThis.navigator,
    IS_REACT_ACT_ENVIRONMENT: (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT,
  }
  const document = new TestDocument()
  setGlobal('document', document)
  setGlobal('window', document.defaultView)
  setGlobal('Event', TestEvent)
  setGlobal('Node', TestNode)
  setGlobal('Element', TestElement)
  setGlobal('HTMLElement', TestElement)
  setGlobal('HTMLInputElement', TestElement)
  setGlobal('HTMLIFrameElement', TestElement)
  setGlobal('HTMLTextAreaElement', TestElement)
  setGlobal('HTMLButtonElement', TestElement)
  setGlobal('navigator', { userAgent: 'vitest' })
  setGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  const container = document.createElement('div')
  document.body.appendChild(container)
  let root: Root
  await act(async () => {
    root = createRoot(container as unknown as Element)
    root.render(render())
  })
  return {
    container,
    document,
    get activeElement() {
      return document.activeElement
    },
    rerender(nextRender = render) {
      return act(async () => root.render(nextRender()))
    },
    cleanup() {
      act(() => root.unmount())
      for (const [name, value] of Object.entries(previousGlobals)) {
        setGlobal(name, value)
      }
    },
  }
}
