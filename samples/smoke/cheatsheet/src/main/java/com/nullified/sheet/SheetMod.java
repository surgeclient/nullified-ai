package com.nullified.sheet;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.itemgroup.v1.ItemGroupEvents;
import net.minecraft.commands.Commands;
import net.minecraft.core.Registry;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.registries.Registries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.server.permissions.Permissions;
import net.minecraft.world.food.FoodProperties;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.CreativeModeTabs;
import net.minecraft.world.item.Item;
import net.minecraft.world.item.Rarity;
import net.minecraft.world.item.ToolMaterial;
import net.minecraft.world.item.equipment.ArmorMaterials;
import net.minecraft.world.item.equipment.ArmorType;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockBehaviour;
import net.minecraft.world.level.material.MapColor;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.ItemTags;

public class SheetMod implements ModInitializer {
	public static final String MOD_ID = "sheet";

	static ResourceKey<Item> itemKey(String name) {
		return ResourceKey.create(Registries.ITEM, Identifier.fromNamespaceAndPath(MOD_ID, name));
	}

	static Item item(String name, Item.Properties props) {
		ResourceKey<Item> key = itemKey(name);
		return Registry.register(BuiltInRegistries.ITEM, key, new Item(props.setId(key)));
	}

	public static final ToolMaterial RUBY_MATERIAL = new ToolMaterial(BlockTags.INCORRECT_FOR_IRON_TOOL, 500, 7f, 2.5f, 15, ItemTags.IRON_TOOL_MATERIALS);
	public static final Item RUBY = item("ruby", new Item.Properties().rarity(Rarity.RARE));
	public static final Item RUBY_SWORD = item("ruby_sword", new Item.Properties().sword(RUBY_MATERIAL, 3f, -2.4f));
	public static final Item IRON_PICK = item("iron_pick", new Item.Properties().pickaxe(ToolMaterial.IRON, 1f, -2.8f));
	public static final Item HELMET = item("helmet", new Item.Properties().humanoidArmor(ArmorMaterials.IRON, ArmorType.HELMET));
	public static final Item SNACK = item("snack", new Item.Properties().food(new FoodProperties.Builder().nutrition(4).saturationModifier(0.3f).build()));

	public static final ResourceKey<Block> COIL_KEY = ResourceKey.create(Registries.BLOCK, Identifier.fromNamespaceAndPath(MOD_ID, "coil"));
	public static final Block COIL = Registry.register(BuiltInRegistries.BLOCK, COIL_KEY,
			new CoilBlock(BlockBehaviour.Properties.of().setId(COIL_KEY).mapColor(MapColor.COLOR_ORANGE).strength(3f, 6f)));

	static {
		ResourceKey<Item> key = ResourceKey.create(Registries.ITEM, COIL_KEY.identifier());
		Registry.register(BuiltInRegistries.ITEM, key, new BlockItem(COIL, new Item.Properties().setId(key).useBlockDescriptionPrefix()));
	}

	@Override
	public void onInitialize() {
		ItemGroupEvents.modifyEntriesEvent(CreativeModeTabs.INGREDIENTS).register(entries -> entries.accept(RUBY));
		CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
				dispatcher.register(Commands.literal("sheet")
						.requires(source -> source.permissions().hasPermission(Permissions.COMMANDS_MODERATOR))
						.executes(ctx -> {
							ctx.getSource().sendSuccess(() -> Component.literal("Done"), false);
							ServerPlayer player = ctx.getSource().getPlayer();
							if (player != null) {
								ServerLevel level = player.level();
								player.sendSystemMessage(Component.literal("You are in " + level.dimension()));
							}
							return 1;
						})));
	}
}
