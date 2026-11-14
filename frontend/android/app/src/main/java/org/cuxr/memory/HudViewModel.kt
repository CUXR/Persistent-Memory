package org.cuxr.memory

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel

class HudViewModel(private val savedState: SavedStateHandle) : ViewModel() {
    val page = savedState.getStateFlow("page", 0)

    var fixtureIndex: Int
        get() = savedState["fixtureIndex"] ?: 0
        set(value) { savedState["fixtureIndex"] = value }

    var factIndex: Int
        get() = savedState["factIndex"] ?: 0
        set(value) { savedState["factIndex"] = value }

    fun advance(direction: Int) {
        savedState["page"] = Math.floorMod(page.value + direction, PAGE_COUNT)
    }

    companion object {
        const val BRIEFING_PAGE = 1
        const val CAMERA_PAGE = 3
        const val MICROPHONE_PAGE = 4
        const val PAGE_COUNT = 5
    }
}
