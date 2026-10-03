package org.cuxr.memory

import android.app.Application
import com.ffalcon.mercury.android.sdk.MercurySDK

class MemoryApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        MercurySDK.init(this)
    }
}
