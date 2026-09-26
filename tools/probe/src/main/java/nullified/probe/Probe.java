package nullified.probe;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.nio.file.Files;
import java.nio.file.Path;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.Item;

/**
 * Runtime check helper: once the dedicated server has fully started with the mod under test,
 * writes every block/item/entity id that isn't vanilla or Fabric to probe.json and stops the server.
 */
public class Probe implements ModInitializer {
	private static boolean ours(Identifier id) {
		String ns = id.getNamespace();
		return !ns.equals("minecraft") && !ns.equals("c") && !ns.startsWith("fabric") && !ns.equals("nullified_probe");
	}

	private static JsonArray ids(Registry<?> registry) {
		JsonArray out = new JsonArray();
		for (Identifier id : registry.keySet()) {
			if (ours(id)) {
				out.add(id.toString());
			}
		}
		return out;
	}

	@Override
	public void onInitialize() {
		ServerLifecycleEvents.SERVER_STARTED.register(server -> {
			JsonObject out = new JsonObject();
			out.add("blocks", ids(BuiltInRegistries.BLOCK));
			out.add("items", ids(BuiltInRegistries.ITEM));
			out.add("entities", ids(BuiltInRegistries.ENTITY_TYPE));
			out.add("block_entities", ids(BuiltInRegistries.BLOCK_ENTITY_TYPE));
			JsonArray blockItems = new JsonArray();
			for (Identifier id : BuiltInRegistries.ITEM.keySet()) {
				Item item = BuiltInRegistries.ITEM.getValue(id);
				if (ours(id) && item instanceof BlockItem blockItem) {
					blockItems.add(id + "=" + BuiltInRegistries.BLOCK.getKey(blockItem.getBlock()));
				}
			}
			out.add("block_items", blockItems);
			try {
				Files.writeString(Path.of("probe.json"), out.toString());
			} catch (Exception e) {
				e.printStackTrace();
			}
			server.halt(false);
		});
	}
}
