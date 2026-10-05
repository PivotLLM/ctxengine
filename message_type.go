/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package ctxengine

// MessageTypeToolError is the Message.Type annotation a host sets on a tool
// result that reports a failure. The eviction sweep reads it so a failed write
// never supersedes the read it needed; the host that runs the tools is the one
// that knows the call failed, so it sets the type when it stores the result.
const MessageTypeToolError = "tool_error"
