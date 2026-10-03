package org.cuxr.memory

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel

class HudViewModel(private val savedState: SavedStateHandle) : ViewModel() {
    val page = savedState.getStateFlow("page", 0)

    fun advance(direction: Int) {
        savedState["page"] = Math.floorMod(page.value + direction, PAGE_COUNT)
    }

    companion object {
        const val PAGE_COUNT = 3
    }
}
