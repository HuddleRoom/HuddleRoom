import { useEffect, useRef, useState } from 'react'
import { useAgentResponseStore } from '@/stores/agent-response'
import { Tabs, TabPanel } from '@/components/common/Tabs'
import { AgentActivityPanel } from './AgentActivityPanel'
import { ProjectChatPanel } from './ProjectChatPanel'

// Ask|Calls tab wrapper for the right rail (P0.1). Default tab: Calls if
// there are active (non-terminal) calls at first mount, else Ask; sticky
// after that — new calls never auto-switch the tab, they only light an
// unread dot on Calls while Ask is active.
export function ProjectRail({ projectId, contextGoalId = null, focusRequest = 0 }: { projectId: string; contextGoalId?: string | null; focusRequest?: number }) {
  const nonterminalCount = useAgentResponseStore((s) => Object.values(s.calls).filter((c) => !c.terminal).length)
  const callCount = useAgentResponseStore((s) => Object.keys(s.calls).length)

  const [activeTab, setActiveTab] = useState<'ask' | 'calls'>(() => (nonterminalCount > 0 ? 'calls' : 'ask'))
  const [callsUnread, setCallsUnread] = useState(false)
  const priorCallCount = useRef(callCount)

  useEffect(() => {
    if (callCount > priorCallCount.current && activeTab !== 'calls') setCallsUnread(true)
    priorCallCount.current = callCount
  }, [callCount, activeTab])

  useEffect(() => { if (focusRequest) setActiveTab('ask') }, [focusRequest])

  const tabs = [
    { id: 'ask', label: 'Ask' },
    { id: 'calls', label: callsUnread ? 'Calls •' : 'Calls' },
  ]

  return <div className="flex h-full flex-col">
    <Tabs
      tabs={tabs}
      activeId={activeTab}
      onChange={(id) => {
        setActiveTab(id as 'ask' | 'calls')
        if (id === 'calls') setCallsUnread(false)
      }}
      idPrefix="rail"
      ariaLabel="Rail content"
      className="mb-2 shrink-0"
    />
    <TabPanel tabId="ask" activeId={activeTab} idPrefix="rail" className="min-h-0 flex-1 overflow-y-auto">
      <ProjectChatPanel projectId={projectId} contextGoalId={contextGoalId} focusRequest={focusRequest} active={activeTab === 'ask'} />
    </TabPanel>
    <TabPanel tabId="calls" activeId={activeTab} idPrefix="rail" className="min-h-0 flex-1">
      <AgentActivityPanel projectId={projectId} />
    </TabPanel>
  </div>
}
