use bevy::{
	app::PluginGroupBuilder,
	prelude::*,
	window::{close_on_esc, WindowMode},
};
use bevy_vulkano::{
	BevyVulkanoContext, BevyVulkanoSettings, BevyVulkanoWindows, VulkanoWinitPlugin,
};

mod render;

pub struct PluginBundle;

impl PluginGroup for PluginBundle {
	fn build(self) -> PluginGroupBuilder {
		PluginGroupBuilder::start::<PluginBundle>()
			.add(bevy::input::InputPlugin)
			.add(bevy::window::WindowPlugin::default())
			.add(VulkanoWinitPlugin)
	}
}

fn main() {
	App::new()
		.insert_non_send_resource(BevyVulkanoSettings {
			is_gui_overlay: true,
			..BevyVulkanoSettings::default()
		})
		.add_plugins(PluginBundle.set(WindowPlugin {
			primary_window: Some(Window {
				resolution: (1920.0, 1080.0).into(),
				present_mode: bevy::window::PresentMode::Fifo,
				resizable: true,
				mode: WindowMode::Windowed,
				..default()
			}),
			..default()
		}))
		.add_systems(Startup, create_pipelines)
		.add_systems(Update, close_on_esc)
		.add_systems(PostUpdate, main_render_system_primary_window)
		.run();
}

fn create_pipelines(
	mut commands: Commands,
	window_query: Query<Entity, With<Window>>,
	context: Res<BevyVulkanoContext>,
	windows: NonSend<BevyVulkanoWindows>,
) {
	let window_entity = window_query.single();
	let primary_window = windows.get_vulkano_window(window_entity).unwrap();

	let render = render::Render::new(
		context.context.memory_allocator().clone(),
		primary_window.renderer.graphics_queue(),
		primary_window.renderer.swapchain_format(),
	);
	commands.insert_resource(render);
}

pub fn main_render_system_primary_window(
	window_query: Query<Entity, With<Window>>,
	mut vulkano_windows: NonSendMut<BevyVulkanoWindows>,
	mut render: ResMut<render::Render>,
) {
	if let Ok(window_entity) = window_query.get_single() {
		let primary_window = vulkano_windows
			.get_vulkano_window_mut(window_entity)
			.unwrap();

		// Start frame
		let before = match primary_window.renderer.acquire() {
			Err(e) => {
				bevy::log::error!("Failed to start frame: {}", e);
				return;
			}
			Ok(f) => f,
		};

		let final_image = primary_window.renderer.swapchain_image_view();
		let after_render = render.render(before, final_image);

		// Finish Frame
		primary_window.renderer.present(after_render, true);
	}
}
