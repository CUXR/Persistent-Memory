package org.cuxr.memory

import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.ffalcon.mercury.android.sdk.touch.TempleAction
import com.ffalcon.mercury.android.sdk.ui.activity.BaseMirrorActivity
import kotlinx.coroutines.launch
import org.cuxr.memory.databinding.ActivityHudBinding

class HudActivity : BaseMirrorActivity<ActivityHudBinding>() {
    private lateinit var hud: HudViewModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        hud = ViewModelProvider(this)[HudViewModel::class.java]
        lifecycleScope.launch {
            repeatOnLifecycle(Lifecycle.State.RESUMED) {
                launch { hud.page.collect(::render) }
                launch {
                    templeActionViewModel.state.collect { action ->
                        if (!action.consumed) {
                            when (action) {
                                is TempleAction.Click,
                                is TempleAction.SlideForward,
                                is TempleAction.SlideDownwards -> hud.advance(1)
                                is TempleAction.SlideBackward,
                                is TempleAction.SlideUpwards -> hud.advance(-1)
                                is TempleAction.DoubleClick -> finish()
                                else -> Unit
                            }
                            // The SDK retains its last event across lifecycle restarts.
                            action.consumed = true
                        }
                    }
                }
            }
        }
    }

    private fun render(page: Int) {
        val titles = resources.getStringArray(R.array.hud_titles)
        val messages = resources.getStringArray(R.array.hud_messages)
        mBindingPair.updateView {
            title.text = titles[page]
            message.text = messages[page]
            pageNumber.text = getString(R.string.page_number, page + 1, HudViewModel.PAGE_COUNT)
        }
        Log.i("MemoryHud", "Rendered page ${page + 1} in both eyes")
    }
}
