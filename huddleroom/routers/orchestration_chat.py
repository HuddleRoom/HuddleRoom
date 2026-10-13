import uuid
from fastapi import APIRouter, Depends, HTTPException
from huddleroom.dependencies import get_current_user
from huddleroom.models.user import User
from huddleroom.schemas.orchestration_chat import ChatSubmitRequest, ChatTurnResponse, ChatHistoryResponse
from huddleroom.services.orchestration_chat_service import OrchestrationChatService
from huddleroom.services.orchestration_conversation_service import ConversationDomainError

router = APIRouter()
chat_service = OrchestrationChatService()


def _error(error):
    return HTTPException(error.status_code, detail={"code": error.code, "message": error.message})


@router.post("/chat", response_model=ChatTurnResponse)
async def submit_chat(project_id: uuid.UUID, data: ChatSubmitRequest, user: User = Depends(get_current_user)):
    try:
        return await chat_service.submit(project_id, user.id, data)
    except ConversationDomainError as error:
        raise _error(error) from error


@router.get("/chat", response_model=ChatHistoryResponse)
async def chat_history(project_id: uuid.UUID, user: User = Depends(get_current_user)):
    try:
        return await chat_service.history(project_id, user.id)
    except ConversationDomainError as error:
        raise _error(error) from error
