package org.cuxr.memory

import com.ffalcon.mercury.android.sdk.touch.TempleAction

class TempleInputAdapter(
    private val next: (Int) -> Unit,
    private val tap: () -> Unit,
    private val exit: () -> Unit,
    private val hold: () -> Unit,
) {
    fun accept(action: TempleAction) {
        if (action.consumed) return
        // Consume before dispatch; paired rendering must never dispatch device actions.
        action.consumed = true
        when (action) {
            is TempleAction.Click -> tap()
            is TempleAction.SlideForward, is TempleAction.SlideDownwards -> next(1)
            is TempleAction.SlideBackward, is TempleAction.SlideUpwards -> next(-1)
            is TempleAction.DoubleClick -> exit()
            is TempleAction.LongClick -> hold()
            else -> Unit
        }
    }
}
