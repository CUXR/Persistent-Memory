package org.cuxr.memory

import android.content.res.AssetManager
import org.json.JSONObject

enum class BriefingMode(val label: Int) {
    HIDDEN(R.string.briefing_hidden),
    NAME(R.string.briefing_name),
    FACTS(R.string.briefing_name_facts);

    fun next() = entries[(ordinal + 1) % entries.size]
}

data class Briefing(val name: String?, val facts: List<String>)

class BriefingFixtures(private val assets: AssetManager) {
    private fun body(file: String): JSONObject =
        assets.open("briefings/$file.json").bufferedReader().use {
            JSONObject(it.readText()).getJSONObject("response").getJSONObject("body")
        }

    private val cases = listOf("brief", "brief_no_memories", "brief_unnamed")

    val count get() = cases.size

    fun load(index: Int): Briefing {
        val brief = body(cases[index])
        val personId = brief.getString("person_id")
        val name = if (brief.isNull("display_name")) null else brief.getString("display_name")
        val hobbies = brief.getJSONObject("hobbies")
        val facts = buildList {
            addFacts(hobbies)
            if (!hobbies.isNull("next_cursor")) {
                val nextPage = body("facts_next_page")
                check(nextPage.getString("person_id") == personId)
                check(nextPage.getString("category") == "hobby")
                addFacts(nextPage)
            }
            addFacts(brief.getJSONObject("affiliations"))
        }
        return Briefing(name, facts)
    }

    private fun MutableList<String>.addFacts(page: JSONObject) {
        val items = page.getJSONArray("items")
        for (index in 0 until items.length()) add(items.getJSONObject(index).getString("text"))
    }
}
